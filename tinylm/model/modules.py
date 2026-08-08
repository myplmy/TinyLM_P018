"""RMSNorm, RoPE, Attention(QK-norm), MLP, Layer."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import TLinear, LoRA

# ---- 어텐션 컴포넌트 레지스트리 (config.attn_kind 로 선택) ----
ATTENTION_KINDS = {}

def register_attention(name):
    def deco(cls):
        ATTENTION_KINDS[name] = cls
        return cls
    return deco

def build_attention(cfg, owns_kv):
    kind = getattr(cfg, "attn_kind", "softmax_cla")
    if kind not in ATTENTION_KINDS:
        raise NotImplementedError(
            f"attn_kind={kind!r} 미구현. 등록됨: {list(ATTENTION_KINDS)}. "
            f"새 어텐션은 @register_attention('name') 로 modules.py 에 추가하세요.\n"
            f"주의: transformer.forward 의 KV-bank(compute_kv/owner) 로직은 softmax_cla 전용이라, "
            f"KDA 등 상태가 다른 종류는 그 오케스트레이션도 함께 일반화해야 합니다(P004).")
    return ATTENTION_KINDS[kind](cfg, owns_kv)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__(); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def build_rope(hd, max_len, theta, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device).float() / hd))
    f = torch.outer(torch.arange(max_len, device=device).float(), inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    x1, x2 = x.float().chunk(2, dim=-1)
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1).to(x.dtype)


@register_attention("softmax_cla")
class Attention(nn.Module):
    """softmax + GQA + CLA + QK-norm. Q/O는 층별 독립, K/V는 cla_group 첫 층만 소유·재사용.
    v5: q·k 내적 전에 파라미터 없는 RMSNorm(QK-norm) 으로 로짓 폭주를 막는다."""

    def __init__(self, cfg, owns_kv: bool):
        super().__init__()
        self.cfg, self.owns_kv = cfg, owns_kv
        o_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        self.q_proj = TLinear(cfg, cfg.dim, cfg.dim, mode_delta=True)
        self.o_proj = TLinear(cfg, cfg.dim, cfg.dim, out_scale=o_scale, mode_delta=True)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        if owns_kv:
            self.k_proj = TLinear(cfg, cfg.dim, cfg.kv_dim)
            self.v_proj = TLinear(cfg, cfg.dim, cfg.kv_dim)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def compute_kv(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg
        k = self.k_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        k = self.k_norm(k)                          # v5: RoPE 전에 정규화
        return apply_rope(k, cos, sin), v

    def forward(self, x, kv, cos, sin, mode_p):
        B, T, _ = x.shape
        c = self.cfg
        q = self.q_proj(x, mode_p).view(B, T, c.n_q_heads, c.head_dim).transpose(1, 2)
        q = self.q_norm(q)                          # v5: RoPE 전에 정규화
        q = apply_rope(q, cos, sin)
        k, v = kv
        n_rep = c.n_q_heads // c.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        # ★KV 캐시가 있으면 q_len < kv_len 이다. 이때 `is_causal=True` 를 그대로 쓰면
        #   SDPA 가 마스크를 **좌상단 정렬**해서 완전히 틀린 위치를 가린다(조용히 틀린다).
        #   질의는 절대위치 [past_len, past_len+q_len) 에 있으므로 tril(past_len) 이 맞다.
        q_len, kv_len = q.shape[2], k.shape[2]
        if q_len == kv_len:                                   # 캐시 없음(=prefill 포함)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            past_len = kv_len - q_len
            mask = torch.ones(q_len, kv_len, dtype=torch.bool,
                              device=q.device).tril(past_len)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(o.transpose(1, 2).contiguous().view(B, T, c.dim), mode_p)


class MLP(nn.Module):
    """중간층에서는 mlp_group개 층이 이 인스턴스 하나를 공유한다."""

    def __init__(self, cfg):
        super().__init__()
        o_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        self.gate_proj = TLinear(cfg, cfg.dim, cfg.ffn_dim, mode_delta=True)
        self.up_proj = TLinear(cfg, cfg.dim, cfg.ffn_dim, mode_delta=True)
        self.down_proj = TLinear(cfg, cfg.ffn_dim, cfg.dim, out_scale=o_scale, mode_delta=True)

    def forward(self, x, mode_p, lora=None, film=None):
        g = self.gate_proj(x, mode_p)
        u = self.up_proj(x, mode_p)
        if lora is not None:                        # 층별 LoRA 보정(gate/up/down)
            g = g + lora[0](x)
            u = u + lora[1](x)
        h = F.silu(g) * u
        if film is not None:                        # 층별 FiLM: 공유 MLP 은닉을 층마다 변조(거의 공짜)
            h = h * (1.0 + film[0]) + film[1]
        d = self.down_proj(h, mode_p)
        if lora is not None:
            d = d + lora[2](h)
        return d


class Layer(nn.Module):
    """어텐션·정규화·게이트는 층 소유. MLP는 참조(공유 가능)."""

    def __init__(self, cfg, owns_kv: bool, mlp: MLP, mlp_lora: bool = False, mlp_film: bool = False):
        super().__init__()
        self.ln1, self.ln2 = RMSNorm(cfg.dim, cfg.norm_eps), RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = build_attention(cfg, owns_kv)
        self.mlp = [mlp]                       # 모듈 등록 회피: 파라미터 중복 계수 방지
        self.has_lora = mlp_lora and cfg.mlp_lora_rank > 0
        if self.has_lora:                      # 공유 MLP를 층별로 특화시키는 저랭크 보정
            r = cfg.mlp_lora_rank
            self.lora_gate = LoRA(cfg, cfg.dim, cfg.ffn_dim, r)
            self.lora_up   = LoRA(cfg, cfg.dim, cfg.ffn_dim, r)
            self.lora_down = LoRA(cfg, cfg.ffn_dim, cfg.dim, r)
        self.has_film = mlp_film and getattr(cfg, "mlp_film", False)
        if self.has_film:                      # 층별 FiLM 파라미터(스케일/시프트, ffn_dim)
            self.film_scale = nn.Parameter(torch.zeros(cfg.ffn_dim))
            self.film_shift = nn.Parameter(torch.zeros(cfg.ffn_dim))
        self.a_scale = nn.Parameter(torch.zeros(cfg.dim))
        self.a_shift = nn.Parameter(torch.zeros(cfg.dim))
        self.m_scale = nn.Parameter(torch.zeros(cfg.dim))
        self.m_shift = nn.Parameter(torch.zeros(cfg.dim))
        self.use_mode_ln = cfg.n_modes > 1
        if self.use_mode_ln:
            self.mode_scale = nn.Parameter(torch.zeros(cfg.n_modes, 2, cfg.dim))
        g0 = 1.0 / math.sqrt(cfg.n_layers)
        self.gates = nn.Parameter(torch.tensor([g0, g0]))

    def forward(self, x, kv, cos, sin, mode_p):
        ms = None
        if self.use_mode_ln and mode_p is not None:
            ms = (mode_p @ self.mode_scale.flatten(1)).view(*mode_p.shape[:2], 2, -1)
        a = (1 + self.a_scale) if ms is None else (1 + self.a_scale + ms[..., 0, :])
        m = (1 + self.m_scale) if ms is None else (1 + self.m_scale + ms[..., 1, :])
        x = x + self.gates[0] * self.attn(self.ln1(x) * a + self.a_shift, kv, cos, sin, mode_p)
        lora = (self.lora_gate, self.lora_up, self.lora_down) if self.has_lora else None
        film = (self.film_scale, self.film_shift) if self.has_film else None
        x = x + self.gates[1] * self.mlp[0](self.ln2(x) * m + self.m_shift, mode_p, lora, film)
        return x
