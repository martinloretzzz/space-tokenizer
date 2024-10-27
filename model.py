# From https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import pack_token, unpack_token


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

    def __init__(self, config, TRAIN_SPACE):
        super().__init__()
        self.config = config
        self.TRAIN_SPACE = TRAIN_SPACE

        space_n_embed = config.n_embd // 8 # reduced dimension for space and upper to save parameters

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        if self.TRAIN_SPACE:
            # slight hack, we use an Embedding layer to lookup the relevant row of the weights
            self.lm_reduce = nn.Linear(config.n_embd, space_n_embed)
            self.lm_space = nn.Embedding(config.vocab_size, space_n_embed)
            self.lm_case = nn.Embedding(config.vocab_size, space_n_embed)

            self.emb_space = nn.Embedding(2, config.n_embd)
            self.emb_case = nn.Embedding(2, config.n_embd)

        # init params
        self.apply(self._init_weights)
        # TODO does normal init work equally well?
        if self.TRAIN_SPACE:
            torch.nn.init.normal_(self.lm_reduce.weight, mean=0.0, std=0.002)
            torch.nn.init.normal_(self.lm_space.weight, mean=0.0, std=0.002)
            torch.nn.init.normal_(self.lm_case.weight, mean=0.0, std=0.002)

            torch.nn.init.normal_(self.emb_space.weight, mean=0.0, std=0.00002)
            torch.nn.init.normal_(self.emb_case.weight, mean=0.0, std=0.00002)

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
        if self.TRAIN_SPACE:
            ids, space, case = unpack_token(idx)
            tok_emb = self.transformer.wte(ids) + self.emb_space(space) + self.emb_case(case)
        else:
            tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        self.hook = None
        loss_out, acc_out, reduced_embed = None, None, None

        if self.TRAIN_SPACE:
            x_grad_scaler = x.clone()
            if x.requires_grad:
                if self.hook is not None:
                    raise Exception("Hook still exists")
                self.hook = x_grad_scaler.register_hook(lambda grad: grad * 0.01) 
            reduced_embed = self.lm_reduce(x_grad_scaler)
        
        logits = self.lm_head(x)

        if targets is not None:
            if self.TRAIN_SPACE:
                targets_ids, targets_space, targets_case = unpack_token(targets)
                logits_space_at_target, logits_case_at_target = self.get_space_case_logits_at(reduced_embed, targets_ids)

                loss_ids = F.cross_entropy(logits.view(-1, logits.size(-1)), targets_ids.view(-1))
                loss_space = F.binary_cross_entropy_with_logits(logits_space_at_target.view(-1), targets_space.view(-1).to(torch.bfloat16))
                loss_case = F.binary_cross_entropy_with_logits(logits_case_at_target.view(-1), targets_case.view(-1).to(torch.bfloat16))

                loss = loss_ids + loss_space + loss_case
                loss_out = (loss, loss_ids, loss_space, loss_case)

                def accuracy(x, y): return (x == y).sum() / y.numel()

                y_ids = torch.argmax(logits, dim=-1)
                y_space = torch.where(logits_space_at_target > 0, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long)).squeeze(-1)
                y_case = torch.where(logits_case_at_target > 0, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long)).squeeze(-1)
                y_token = pack_token(y_ids, y_space, y_case)

                acc_out = (accuracy(y_token, targets), accuracy(y_ids, targets_ids), accuracy(y_space, targets_space), accuracy(y_case, targets_case))
            else:
                k = 4
                topk_vals, topk_indices = torch.topk(targets, k, dim=-1)

                logits_reshape = logits.view(-1, logits.size(-1))
                loss_ids_0 = torch.mean(topk_vals[:,:,0].view(-1) * F.cross_entropy(logits_reshape, topk_indices[:,:,0].view(-1), reduction='none'))
                loss_ids_1 = torch.mean(topk_vals[:,:,1].view(-1)* F.cross_entropy(logits_reshape, topk_indices[:,:,1].view(-1), reduction='none'))
                loss_ids_2 = torch.mean(topk_vals[:,:,2].view(-1) * F.cross_entropy(logits_reshape, topk_indices[:,:,2].view(-1), reduction='none'))
                loss_ids_3 = torch.mean(topk_vals[:,:,3].view(-1) * F.cross_entropy(logits_reshape, topk_indices[:,:,3].view(-1), reduction='none'))
                loss_ids = (loss_ids_0 + loss_ids_1 + loss_ids_2 + loss_ids_3) / 4

                numel = targets.shape[0] * targets.shape[1] # B * T
                acc_ids = (torch.argmax(logits, dim=-1) == torch.argmax(targets, dim=-1)).sum() / numel
                zero = torch.tensor(0, device=targets.device)
                loss_out = (loss_ids, loss_ids, zero, zero)
                acc_out = (acc_ids, acc_ids, zero, zero)

        return logits, loss_out, acc_out, reduced_embed

    def remove_hooks(self):
        if self.hook is not None:
            self.hook.remove()

    def get_space_case_logits_at(self, x, ids):
        logits_space_weight = self.lm_space(ids)
        logits_case_weight = self.lm_case(ids)

        assert x.shape == logits_space_weight.shape
        logits_space = (x * logits_space_weight).sum(-1)
        logits_case = (x * logits_case_weight).sum(-1)
        return logits_space.unsqueeze(-1), logits_case.unsqueeze(-1)

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

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
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