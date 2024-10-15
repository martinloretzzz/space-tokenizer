# sudo apt-get install unzip
# pip install transformers wandb tiktoken
# unzip /workspace/fineweb-25k.zip -d /workspace/dataset-25K/
# unzip /workspace/fineweb-ref.zip -d /workspace/dataset-ref/
# torchrun --standalone --nproc_per_node=4 train-gpt2.py

import inspect
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F

from tokenizer import SpaceTokenizer

TRAIN_SPACE = True
data_root = "dataset-25K/content/data/" if TRAIN_SPACE else "dataset-ref/content/data/"

total_batch_size = 294912 # 262144 # 294912 # 491520 # 524288 # 2**19, ~0.5M, in number of tokens
B = 24 # 48 # 96 if TRAIN_SPACE else 80 # 64 # micro batch size # 64
T = 1024 # sequence length

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 500 # 715
max_steps = 5000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

checkpoint_path = None # "model_03000.pt" # None

vocab_size = (25000 + 257) if TRAIN_SPACE else 50257

with open('./tokenizer-space.json', 'r', encoding='utf-8') as f: tokenizer_config = json.load(f)
tokenizer = SpaceTokenizer(tokenizer_config["model"]["vocab"], vocab_size)

if not TRAIN_SPACE:
    tokenizer = tiktoken.get_encoding("gpt2")

with open('wandb.txt', 'r') as file:
    wandb_key = file.read()
wandb.login(key=wandb_key)

# From https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        non_space_n_embed = (config.n_embd - 2) if TRAIN_SPACE else config.n_embd

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, non_space_n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(non_space_n_embed),
        ))

        self.lm_head = nn.Linear(non_space_n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        if TRAIN_SPACE:
            # slight hack, we use an Embedding layer to lookup the relevant row of the weights
            self.lm_space = nn.Embedding(config.vocab_size, config.n_embd)
            self.lm_case = nn.Embedding(config.vocab_size, config.n_embd)

        # init params
        self.apply(self._init_weights)
        # TODO does normal init work equally well?
        if TRAIN_SPACE:
            torch.nn.init.normal_(self.lm_space.weight, mean=0.0, std=0.002)
            torch.nn.init.normal_(self.lm_case.weight, mean=0.0, std=0.002)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)

        # token embeddings of shape (B, T, n_embd)
        if TRAIN_SPACE:
            # give space and case the last 2 dimensions of embed
            ids, space, upper = self.unpack_token(idx)
            space = 0.1 * (space.unsqueeze(-1).to(torch.bfloat16) - 0.5) # either -0.05 and +0.05
            case = 0.1 * (upper.unsqueeze(-1).to(torch.bfloat16) - 0.5)

            tok_emb = torch.cat([self.transformer.wte(ids), space, case], dim=-1)
        else:
            tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        x_f = x

        if TRAIN_SPACE:
            x = x[:, :, 0:self.config.n_embd-2] # remove space and upper from embed
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss_out, acc_out = None, None
        if targets is not None:
            if TRAIN_SPACE:
                targets_ids, targets_space, targets_case = self.unpack_token(targets)
                logits_space_at_target, logits_case_at_target = self.get_space_case_logits_at(x_f, targets_ids)

                loss_ids = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_ids.view(-1))
                loss_space = F.binary_cross_entropy_with_logits(logits_space_at_target.view(-1), targets_space.view(-1).to(torch.bfloat16))
                loss_case = F.binary_cross_entropy_with_logits(logits_case_at_target.view(-1), targets_case.view(-1).to(torch.bfloat16))

                loss = loss_ids + 0.0001 * loss_space + 0.0001 * loss_case
                loss_out = (loss, loss_ids, loss_space, loss_case)

                def accuracy(x, y): return (x == y).sum() / y.numel()

                y_ids = torch.argmax(logits, dim=-1)
                y_space = torch.where(logits_space_at_target > 0, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long)).squeeze(-1)
                y_case = torch.where(logits_case_at_target > 0, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long)).squeeze(-1)
                y_token = self.pack_token(y_ids, y_space, y_case)

                acc_out = (accuracy(y_token, targets), accuracy(y_ids, targets_ids), accuracy(y_space, targets_space), accuracy(y_case, targets_case))
            else:
                loss_ids = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                acc_ids = (torch.argmax(logits, dim=-1) == targets_ids).sum() / targets.view(-1).size(0)
                loss_out = (loss_ids, loss_ids, torch.tensor(0), torch.tensor(0))
                acc_out = (acc_ids, acc_ids, torch.tensor(0), torch.tensor(0))

        return logits, loss_out, acc_out, x_f

    def get_space_case_logits_at(self, x, ids):
        logits_space_weight = self.lm_space(ids)
        logits_case_weight = self.lm_case(ids)

        assert x.shape == logits_space_weight.shape
        logits_space = (x * logits_space_weight).sum(-1)
        logits_case = (x * logits_case_weight).sum(-1)
        return logits_space.unsqueeze(-1), logits_case.unsqueeze(-1)

    def unpack_token(self, token):
        id = token >> 2
        space = (token >> 1) & 0x01
        upper = (token >> 0) & 0x01
        return id, space, upper

    def pack_token(self, id, space, upper):
        return (id << 2) + (space << 1) + (upper << 0)

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train-gpt2.py

import torch.distributed as dist
# run the training loop
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=vocab_size))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2

start_step = 0
if checkpoint_path is not None:
    loaded = torch.load(checkpoint_path, weights_only=False)
    start_step = loaded["step"]
    model_checkpoint = loaded["model"]
    model.load_state_dict(model_checkpoint)
    print(f"Load model from {checkpoint_path}. Continue at step {start_step}.")

model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# Test if model throws any errors
x, y = val_loader.next_batch()
print(x.shape, y.shape)

x, y = train_loader.next_batch()
x, y = x.to(device), y.to(device)
with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    logits, loss, acc, embed_f = model(x, y)
print(acc)


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

if master_process:
    project_name = f"x-{'25K' if vocab_size < 26000 else '50K'}" if TRAIN_SPACE else "x-ref"
    run = wandb.init(
        project="space-gpt",
        name=project_name,
        config={
            "vocab_size": vocab_size,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "total_batch_size": total_batch_size,
            "B": B,
            "T": T
        },
    )

for step in range(start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 25 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            loss_accum = [0.0, 0.0, 0.0, 0.0]
            acc_accum = [0.0, 0.0, 0.0, 0.0]
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, losses, acces, _ = model(x, y)
    
                for i in range(4):
                    loss_accum[i] += (losses[i] / grad_accum_steps).detach()
                    acc_accum[i] += (acces[i] / val_loss_steps).detach()
        if ddp:
            for i in range(4):
                dist.all_reduce(loss_accum[i], op=dist.ReduceOp.AVG)
                dist.all_reduce(acc_accum[i], op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {loss_accum[0].item():.4f}, ids: {loss_accum[1].item():.4f}, space: {loss_accum[2].item():.4f}, case: {loss_accum[3].item():.4f} | acc: {acc_accum[0].item():.2f} | acc_ids: {acc_accum[1].item():.2f} | acc_space: {acc_accum[2].item():.2f} | acc_case: {acc_accum[3].item():.2f}")
            run.log({"step": step, "val/loss": loss_accum[0].item(), "val/loss_ids": loss_accum[1].item(), "val/loss_space": loss_accum[2].item(), "val/loss_case": loss_accum[3].item(), "val/acc": acc_accum[0].item(), "val/acc_ids": acc_accum[1].item(), "val/acc_space": acc_accum[2].item(), "val/acc_case": acc_accum[3].item()})

            if step > 0 and (step % 1000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = f"model_{step:05d}.pt"
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': loss_accum[0].item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

                artifact = wandb.Artifact(name=f"{project_name}-model{'-full' if max_steps > 10000 else ''}", type="model")
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 100 == 0) or last_step) and (not use_compile):
        samples = []

        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = tokenizer.encode("The brown dog")

        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _, __, embed_f = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]

                # get the probabilities
                probs = F.softmax(logits, dim=-1)

                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    
                if TRAIN_SPACE:
                    logits_space_at_target, logits_case_at_target = raw_model.get_space_case_logits_at(embed_f[:, -1, :], xcol.squeeze(-1))
                    space = torch.where(logits_space_at_target > 0, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long))
                    upper = torch.where(logits_case_at_target > 0, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long))
                    xcol = raw_model.pack_token(xcol, space, upper)
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            samples.append(decoded)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
        samples = "\n".join(samples)
        if master_process:
            run.log({"step": step, "samples": samples})

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum, acc = [0.0, 0.0, 0.0, 0.0], 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, losses, acces, ___ = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        acc += acces[0] / grad_accum_steps
        for i, loss in enumerate(losses):
            loss_accum[i] += (loss / grad_accum_steps).detach()
        loss = losses[0] / grad_accum_steps
        loss.backward()
    if ddp:
        for accum in loss_accum:
            dist.all_reduce(accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        if step % 10 == 0:
            print(f"step {step:5d} | loss: {loss_accum[0].item():.6f} | acc: {acc:.2f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        run.log({"step": step, "loss": loss_accum[0].item(), "loss_ids": loss_accum[1].item(), "loss_space": loss_accum[2].item(), "loss_case": loss_accum[3].item(), "lr":lr, "norm":norm, "dt":1000*dt, "tokens_per_sec": tokens_per_sec, "acc": acc })

if master_process:
    run.finish()

if ddp:
    destroy_process_group()
