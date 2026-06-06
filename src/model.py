from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FuncT = TypeVar("FuncT", bound=Callable[..., object])


def disable_torch_compile(fn: FuncT) -> FuncT:
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if disable is None:
        dynamo = getattr(torch, "_dynamo", None)
        disable = getattr(dynamo, "disable", None)
    if disable is None:
        return fn
    return disable(fn)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 16
    n_embd: int = 512
    n_head: int = 8
    n_kv_head: int = 4
    dropout: float = 0.0
    bias: bool = False
    use_gradient_checkpointing: bool = False
    attention_mode: str = "full"
    hybrid_attention_pattern: str = "full,sliding,csa,hca,sliding,csa,hca,full"
    sliding_window: int = 512
    csa_block_size: int = 64
    csa_local_window: int = 512
    hca_block_size: int = 256
    hca_local_window: int = 512
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, D], cos/sin: [1, T, 1, D]
    return (x * cos) + (rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Resolve the attention kind once at construction. layer_idx and config are
        # fixed for the lifetime of the module, so this never changes per forward.
        # Computing it inside forward() makes torch.compile treat layer_idx as a
        # dynamic guard and recompile per layer until it hits recompile_limit and
        # falls the whole frame back to eager (projections included). Caching it as
        # a constant lets each layer compile to its own static graph.
        self._kind = self._attention_kind()

    def _rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, :, None, :]
        sin = emb.sin().to(dtype=dtype)[None, :, None, :]
        return cos, sin

    def _attention_kind(self) -> str:
        mode = self.config.attention_mode.lower().strip()
        if mode == "hybrid":
            pattern = [
                part.strip().lower()
                for part in self.config.hybrid_attention_pattern.split(",")
                if part.strip()
            ]
            if not pattern:
                raise ValueError("hybrid_attention_pattern must contain at least one attention mode.")
            mode = pattern[self.layer_idx % len(pattern)]
        aliases = {
            "global": "full",
            "compressed": "csa",
            "compressed_sparse": "csa",
            "heavily_compressed": "hca",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"full", "sliding", "csa", "hca"}:
            raise ValueError(f"Unsupported attention mode: {mode}")
        return mode

    def _local_mask(self, seq_len: int, window: int, device: torch.device) -> torch.Tensor:
        window = max(1, int(window))
        q_pos = torch.arange(seq_len, device=device)[:, None]
        k_pos = torch.arange(seq_len, device=device)[None, :]
        return (k_pos <= q_pos) & (k_pos > q_pos - window)

    def _summary_positions(self, seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
        n_blocks = (seq_len + block_size - 1) // block_size
        block_ids = torch.arange(n_blocks, device=device)
        return torch.clamp((block_ids + 1) * block_size - 1, max=seq_len - 1)

    def _block_summaries(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        B, H, T, D = x.shape
        block_size = max(1, min(int(block_size), T))
        n_blocks = (T + block_size - 1) // block_size
        pad_tokens = n_blocks * block_size - T
        if pad_tokens:
            x = F.pad(x, (0, 0, 0, pad_tokens))
        x = x.view(B, H, n_blocks, block_size, D)
        counts = torch.full((n_blocks,), block_size, device=x.device, dtype=x.dtype)
        if pad_tokens:
            counts[-1] = block_size - pad_tokens
        return x.sum(dim=3) / counts.view(1, 1, n_blocks, 1)

    @disable_torch_compile
    def _sliding_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self._local_mask(seq_len, self.config.sliding_window, device),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    @disable_torch_compile
    def _compressed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        local_window: int,
    ) -> torch.Tensor:
        T = q.size(2)
        block_size = max(1, min(int(block_size), T))
        local_window = max(1, min(int(local_window), T))

        mem_k = self._block_summaries(k, block_size)
        mem_v = self._block_summaries(v, block_size)
        k_cat = torch.cat((mem_k, k), dim=2)
        v_cat = torch.cat((mem_v, v), dim=2)

        q_pos = torch.arange(T, device=q.device)[:, None]
        mem_pos = self._summary_positions(T, block_size, q.device)[None, :]
        token_pos = torch.arange(T, device=q.device)[None, :]
        mem_visible = mem_pos <= q_pos - local_window
        local_visible = (token_pos <= q_pos) & (token_pos > q_pos - local_window)
        attn_mask = torch.cat((mem_visible, local_visible), dim=1)

        return F.scaled_dot_product_attention(
            q, k_cat, v_cat,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        cos, sin = self._rope_cache(T, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, KVH, T, D]
        v = v.transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        kind = self._kind
        if kind == "full":
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif kind == "sliding":
            if int(self.config.sliding_window) >= T:
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )
            else:
                y = self._sliding_attention(q, k, v, T, x.device)
        elif kind == "csa":
            y = self._compressed_attention(q, k, v, self.config.csa_block_size, self.config.csa_local_window)
        else:
            y = self._compressed_attention(q, k, v, self.config.hca_block_size, self.config.hca_local_window)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # LLaMA-like FFN hidden sizing. 4 * d * 2/3 rounded upward.
        hidden = int(8 * config.n_embd / 3)
        hidden = 256 * ((hidden + 255) // 256)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight  # tied embeddings

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.config.block_size}")

        x = self.dropout(self.tok_embeddings(idx))
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss_targets = targets.reshape(-1)
            loss_targets = torch.where(loss_targets == -100, torch.full_like(loss_targets, -1), loss_targets)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), loss_targets, ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def _apply_repetition_penalty(self, logits: torch.Tensor, idx: torch.Tensor, penalty: float) -> None:
        if penalty <= 1.0:
            return
        for batch_idx in range(logits.size(0)):
            seen = torch.unique(idx[batch_idx])
            token_logits = logits[batch_idx, seen]
            logits[batch_idx, seen] = torch.where(token_logits < 0, token_logits * penalty, token_logits / penalty)

    def _apply_no_repeat_ngram(self, logits: torch.Tensor, idx: torch.Tensor, ngram_size: int) -> None:
        if ngram_size <= 0:
            return
        for batch_idx in range(logits.size(0)):
            tokens = idx[batch_idx].tolist()
            if len(tokens) < ngram_size - 1:
                continue
            prefix = tuple(tokens[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
            banned = []
            for i in range(0, len(tokens) - ngram_size + 1):
                ngram = tuple(tokens[i:i + ngram_size])
                if (ngram_size == 1 and prefix == tuple()) or ngram[:-1] == prefix:
                    banned.append(ngram[-1])
            if banned:
                logits[batch_idx, banned] = -float("inf")

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            self._apply_repetition_penalty(logits, idx, repetition_penalty)
            self._apply_no_repeat_ngram(logits, idx, no_repeat_ngram_size)
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, -float("inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def model_from_dict(cfg: dict) -> GPT:
    return GPT(ModelConfig(**cfg))
