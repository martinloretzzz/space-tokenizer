# sudo apt-get install unzip
# pip install transformers wandb tiktoken
# unzip /workspace/dataset-space-20k-rs.zip -d /workspace/dataset-space/
# unzip /workspace/dataset-ref-20k.zip -d /workspace/dataset-ref/
# torchrun --standalone --nproc_per_node=4 train-gpt2.py
# python train-gpt2.py
# tmux capture-pane -pS -1000000 > log.txt

# dataset-space-20k-rs: 10.09 BT
# dataset-space-50k-rs: 9.59 BT
# dataset-ref-20k: 11.2 BT
# dataset-ref-50k: 10.35 BT
# dataset-ref-50k-tiktoken: 9.95 BT

import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F

from hellaswag import get_most_likely_row, iterate_examples, render_example
from model import GPT, GPTConfig
from tokenizer import (HfTokenizerWrapper, SpaceTokenizer, pack_token,
                       unpack_token)
from tokenizers import Tokenizer

TRAIN_SPACE = False
ENABLE_WANDB = True

data_root = "dataset-space/content/data/" if TRAIN_SPACE else "dataset-ref/content/data/"

total_batch_size = 262144 # 294912@24 262144@16/32 # 294912 # 491520 # 524288 # 2**19, ~0.5M, in number of tokens
B = 8 # 48 # 96 if TRAIN_SPACE else 80 # 64 # micro batch size # 64
T = 1024 # sequence length

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 500 # 715
max_steps = 5000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

checkpoint_path = None

vocab_size = (50000 + 257) #  if TRAIN_SPACE else 50257

project_name = f"lin-{'25K' if vocab_size < 26000 else '50K'}{'-ref' if not TRAIN_SPACE else ''}{'-full' if max_steps > 10000 else ''}"

with open('./tokenizer-space-50k-rs.json', 'r', encoding='utf-8') as f: tokenizer_config = json.load(f)
tokenizer = SpaceTokenizer(tokenizer_config)

if not TRAIN_SPACE:
    # tokenizer = tiktoken.get_encoding("gpt2")
    # tokenizer = HfTokenizerWrapper(Tokenizer.from_file("tokenizer-ref-50k.json"))
    with open('./tokenizer-ref-50k-lin.json', 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)["model"]["vocab"]
        tokenizer_config = {v:k for k, v in tokenizer_config.items()}
    tokenizer = SpaceTokenizer(tokenizer_config, space_expand_vocab=False)
    
with open('wandb.txt', 'r') as file:
    wandb_key = file.read()


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
model = GPT(GPTConfig(vocab_size=vocab_size), TRAIN_SPACE=TRAIN_SPACE)
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
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, master_process=master_process)

if master_process:
    wandb.login(key=wandb_key)
    model_artifact_name = f"{project_name}-model-{random.randint(0, 1000)}"
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
        mode=None if ENABLE_WANDB else "disabled"
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
                    loss_accum[i] += (losses[i] / val_loss_steps).detach()
                    acc_accum[i] += (acces[i] / val_loss_steps).detach()
        if ddp:
            for i in range(4):
                dist.all_reduce(loss_accum[i], op=dist.ReduceOp.AVG)
                dist.all_reduce(acc_accum[i], op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {loss_accum[0].item():.4f}, ids: {loss_accum[1].item():.4f}, space: {loss_accum[2].item():.4f}, case: {loss_accum[3].item():.4f} | acc: {acc_accum[0].item():.2f} | acc_ids: {acc_accum[1].item():.2f} | acc_space: {acc_accum[2].item():.2f} | acc_case: {acc_accum[3].item():.2f}")
            run.log({"step": step, "val/loss": loss_accum[0].item(), "val/loss_ids": loss_accum[1].item(), "val/loss_space": loss_accum[2].item(), "val/loss_case": loss_accum[3].item(), "val/acc": acc_accum[0].item(), "val/acc_ids": acc_accum[1].item(), "val/acc_space": acc_accum[2].item(), "val/acc_case": acc_accum[3].item()})

            if step > 0 and (step % 2000 == 0 or last_step):
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

                artifact = wandb.Artifact(name=model_artifact_name, type="model")
                artifact.add_file(local_path=checkpoint_path)
                run.log_artifact(artifact)


    # once in a while evaluate hellaswag
    if (step % 100 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example, tokenizer)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # print(mask.shape, label)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _, __, ___ = model(tokens)
                if TRAIN_SPACE:
                    tokens = unpack_token(tokens)[0]
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            run.log({"step": step, "val/hellaswag_correct": num_correct_norm, "val/hellaswag_acc": acc_norm})


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
                    xcol = pack_token(xcol, space, upper)
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
        raw_model.remove_hooks()
    if ddp:
        for accum in loss_accum:
            dist.all_reduce(accum, op=dist.ReduceOp.AVG)
        dist.all_reduce(acc, op=dist.ReduceOp.AVG)

    space_emb_norm = 0
    if TRAIN_SPACE:
        space_emb_norm = torch.nn.utils.clip_grad_norm_(raw_model.emb_space.parameters(), 0.0001)
        case_emb_norm = torch.nn.utils.clip_grad_norm_(raw_model.emb_case.parameters(), 0.0001)
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
            print(f"step {step:5d} | loss: {loss_accum[0].item():.6f} | acc: {acc:.2f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | space_norm: {space_emb_norm:0.2f}")
        run.log({"step": step, "loss": loss_accum[0].item(), "loss_ids": loss_accum[1].item(), "loss_space": loss_accum[2].item(), "loss_case": loss_accum[3].item(), "lr":lr, "norm":norm, "dt":1000*dt, "tokens_per_sec": tokens_per_sec, "acc": acc, "space_norm": space_emb_norm })

if master_process:
    run.finish()

if ddp:
    destroy_process_group()
