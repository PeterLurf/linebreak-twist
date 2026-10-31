"""Architecture adapters: blocks, attention input capture, pre-RoPE q/k, RoPE, head output projection."""
from __future__ import annotations

import torch
import torch.nn.functional as F

LLAMA_LIKE = {"gemma2", "gemma", "qwen2", "llama", "mistral"}


class Arch:
    def __init__(self, model):
        self.model = model
        cfg = model.config
        self.type = cfg.model_type
        if self.type in LLAMA_LIKE:
            self.blocks = model.model.layers
            self.H, self.KV = cfg.num_attention_heads, cfg.num_key_value_heads
            self.dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.H
        elif self.type == "gpt_neox":
            self.blocks = model.gpt_neox.layers
            self.H = self.KV = cfg.num_attention_heads
            self.dh = cfg.hidden_size // self.H
        elif self.type == "gpt2":
            self.blocks = model.transformer.h
            self.H = self.KV = cfg.n_head
            self.dh = cfg.n_embd // self.H
        else:
            raise ValueError(f"unsupported model_type {self.type}")
        self.n_layers = len(self.blocks)

    def attn(self, l):
        b = self.blocks[l]
        return {"gpt_neox": lambda: b.attention, "gpt2": lambda: b.attn}.get(self.type, lambda: b.self_attn)()

    def out_proj(self, l):
        a = self.attn(l)
        return {"gpt_neox": lambda: a.dense, "gpt2": lambda: a.c_proj}.get(self.type, lambda: a.o_proj)()

    def capture_attn_inputs(self, layers, store: dict):
        """Pre-hooks recording (normed attention input [T,d], (cos, sin) or None) per layer."""
        hooks = []
        for l in layers:
            def pre(mod, args, kw, l=l):
                x = kw.get("hidden_states", args[0] if args else None)
                store[l] = (x, kw.get("position_embeddings"))
            hooks.append(self.attn(l).register_forward_pre_hook(pre, with_kwargs=True))
        return hooks

    def qk(self, l, x):
        """x: [T, d] float32 normed attention input -> pre-RoPE q [T,H,dh], k [T,KV,dh]."""
        a = self.attn(l)
        T = x.shape[0]
        if self.type in LLAMA_LIKE:
            q = F.linear(x, a.q_proj.weight.float(), None if a.q_proj.bias is None else a.q_proj.bias.float())
            k = F.linear(x, a.k_proj.weight.float(), None if a.k_proj.bias is None else a.k_proj.bias.float())
            return q.view(T, self.H, self.dh), k.view(T, self.KV, self.dh)
        if self.type == "gpt_neox":
            w = a.query_key_value
            qkv = F.linear(x, w.weight.float(), w.bias.float()).view(T, self.H, 3 * self.dh)
            q, k, _ = qkv.chunk(3, dim=-1)
            return q, k
        qkv = x @ a.c_attn.weight.float() + a.c_attn.bias.float()  # Conv1D
        q, k, _ = qkv.split(self.H * self.dh, dim=-1)
        return q.view(T, self.H, self.dh), k.view(T, self.H, self.dh)

    @staticmethod
    def rope(x, pe):
        """Apply (possibly partial) RoPE; x [T, n, dh]; pe = (cos, sin) with shape [1, T, r] or None."""
        if pe is None:
            return x
        cos, sin = pe[0][0].float()[:, None], pe[1][0].float()[:, None]
        r = cos.shape[-1]
        xr, xp = x[..., :r], x[..., r:]
        x1, x2 = xr[..., : r // 2], xr[..., r // 2:]
        xr = xr * cos + torch.cat((-x2, x1), -1) * sin
        return torch.cat((xr, xp), -1)

    def kv_of(self, h):
        return h // (self.H // self.KV)
