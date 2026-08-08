"""커스텀 삼진 학습 커널 (기존 경로와 분리, 기본 off).

설계 원칙
  - **기존 학습 무영향**: cfg.use_ternary_kernel=False(기본)면 이 모듈은 호출되지 않는다.
  - **정확성 우선**: use_ternary_kernel=True 라도 기본은 '레퍼런스'(int8 패킹 + dequant 행렬곱)로,
    기존 `_TernarySTE`+`F.linear` 경로와 **수학적으로 동일**하다. STE backward도 동일 window.
  - **Triton은 검증 후**: cfg.ternary_kernel_triton=True 를 추가로 줘야 Triton forward를 쓴다.
    Triton 커널은 GPU 검증이 끝나기 전엔 켜지 말 것(잘못된 결과가 조용히 학습을 오염시킬 수 있음).
    Triton 미설치/에러 시 자동으로 레퍼런스로 폴백한다.

패킹 표현
  - codes: int8 {-1,0,+1}  [O, I]
  - alpha: 그룹(g128)별 스케일  [O, I//group]
  - 복원 wq[o,i] = codes[o,i] * alpha[o, i//group]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_KERNEL_WARNED = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton forward 커널 (검증 전 사용 금지). x[M,K] @ wq[N,K]^T, wq=codes*alpha(그룹).
# BLOCK_K = group(128) 로 두어 k-블록 하나 = alpha 그룹 하나.
# ---------------------------------------------------------------------------
if _HAS_TRITON:
    @triton.jit
    def _tern_mm_kernel(X, CODES, ALPHA, Y, M, N, K,
                        sxm, sxk, scn, sck, san, sag, sym, syn,
                        BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_g = tl.arange(0, G)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K, G):
            offs_k = kb + offs_g
            x = tl.load(X + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            codes = tl.load(CODES + offs_n[:, None] * scn + offs_k[None, :] * sck,
                            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0)
            alpha = tl.load(ALPHA + offs_n * san + (kb // G) * sag,
                            mask=offs_n < N, other=0.0)
            w = codes.to(tl.float32) * alpha[:, None]          # [BN, G]
            acc += tl.dot(x.to(tl.float32), tl.trans(w))       # [BM,G]@[G,BN]
        y = acc.to(tl.float32)
        tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn, y,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def _triton_matmul(x2d, codes2d, alpha2d, group):
        M, K = x2d.shape
        N = codes2d.shape[0]
        y = torch.empty((M, N), device=x2d.device, dtype=torch.float32)
        BM, BN = 64, 64
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _tern_mm_kernel[grid](
            x2d, codes2d, alpha2d, y, M, N, K,
            x2d.stride(0), x2d.stride(1), codes2d.stride(0), codes2d.stride(1),
            alpha2d.stride(0), alpha2d.stride(1), y.stride(0), y.stride(1),
            BM=BM, BN=BN, G=group)
        return y


def _ref_matmul(x, codes, alpha, group, dtype):
    """레퍼런스: dequant 후 F.linear. 기존 경로와 동일 결과(정확성 기준)."""
    O = codes.shape[0]
    wq = (codes.to(dtype) * alpha.to(dtype)).reshape(O, -1)   # [O, I] (활성 dtype로 통일)
    return F.linear(x, wq)


class _TernaryKernelLinear(torch.autograd.Function):
    """저비트 forward + STE 보존 backward. **삼진 어닐 반영** → 기존 경로와 값·gradient 모두 동일.
    anneal>=1(완전 삼진)일 때만 저비트 커널/레퍼런스 사용, anneal<1은 full-precision 블렌드(값)."""

    @staticmethod
    def forward(ctx, x, w, group, thr, clip, anneal, use_triton):
        O, I = w.shape
        wg = w.reshape(O, I // group, group)
        aw = wg.abs()
        mask = (aw >= thr * aw.mean(dim=2, keepdim=True)).to(w.dtype)
        cnt = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        alpha = (aw * mask).sum(dim=2, keepdim=True) / cnt      # [O, G, 1]
        codes = (torch.sign(wg) * mask).to(torch.int8)          # [O, G, group]
        wq = (codes.to(w.dtype) * alpha).reshape(O, I)          # 삼진 값

        if anneal >= 1.0:                                       # 완전 삼진 → 저비트 경로
            eff = wq
            if use_triton and _HAS_TRITON and x.is_cuda:
                try:
                    x2d = x.reshape(-1, I).contiguous()
                    codes2d = codes.reshape(O, I).contiguous()
                    alpha2d = alpha.reshape(O, I // group).contiguous().float()
                    y = _triton_matmul(x2d, codes2d, alpha2d, group).to(x.dtype)
                    y = y.reshape(*x.shape[:-1], O)
                except Exception as e:                          # 안전 폴백(1회만 경고)
                    global _KERNEL_WARNED
                    if not _KERNEL_WARNED:
                        print(f"[ternary_kernel] Triton 실패 → 레퍼런스 폴백(이후 조용히): {str(e)[:120]}")
                        _KERNEL_WARNED = True
                    y = F.linear(x, wq.to(x.dtype))
            else:
                y = F.linear(x, wq.to(x.dtype))
        else:                                                  # 어닐 중: 기존 STE와 동일 블렌드(값)
            eff = wq + (1.0 - anneal) * (w - wq)
            y = F.linear(x, eff.to(x.dtype))

        ctx.save_for_backward(x, aw, alpha, eff)
        ctx.meta = (O, I, group, clip)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, aw, alpha, eff = ctx.saved_tensors
        O, I, group, clip = ctx.meta
        gx = gy @ eff.to(gy.dtype)                             # off 경로와 동일(값 eff)
        gw_full = gy.reshape(-1, O).t().float() @ x.reshape(-1, I).float()
        win = (1.0 / (1.0 + (aw / (clip * alpha).clamp_min(1e-8)).pow(4))).float()
        gw = (gw_full.reshape(O, I // group, group) * win).reshape(O, I)
        return gx, gw, None, None, None, None, None


def ternary_kernel_linear(x, w, cfg, anneal_t=None):
    """TLinear.forward 진입점. 어닐은 스칼라 텐서(anneal_t)를 우선 사용해 torch.compile 가드를 피한다.
    (anneal_t 미제공 시 cfg.quant_anneal 폴백 — compile 없이 실행하는 경우엔 무해.)"""
    anneal = anneal_t if anneal_t is not None else float(cfg.quant_anneal)
    return _TernaryKernelLinear.apply(
        x, w, cfg.micro_group, cfg.twn_thr_ratio, cfg.ste_clip,
        anneal, getattr(cfg, "ternary_kernel_triton", False))
