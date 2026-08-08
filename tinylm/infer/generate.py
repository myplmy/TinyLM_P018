"""추론기: 체크포인트를 불러와 프롬프트에서 샘플링한다.

P030 단계1(2026-07-31)에서 **KV 캐시 · `<eos>` 정지 · 삼진 1회 계산**을 넣었다.
그전에는 매 스텝 전체 시퀀스를 재계산했고(O(T²)), 매 forward 마다 삼진화를 다시 했으며
(결과 014: CPU 시간의 ~79%), eos 를 무시해 문서 경계를 넘어 생성했다(결과 013).

**`use_cache=False` 는 지우지 말 것** — 캐시 경로의 정확성을 대조하는 유일한 수단이다.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .. import paths
from ..config import TMTConfig
from ..model import TiedMLPTransformer
from ..data import tokenizer_path

CKPT = paths.RUNS / "ckpt"


def _strip(sd):
    # torch.compile 로 저장하면 '_orig_mod.' 접두사가 붙는다.
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_model(arch="tied", ckpt_path=None, device=None, drop_latent=False, int8_store=False,
               unpack_cache=False):
    """`drop_latent=True` 면 P034 단계2 — fp32 latent 를 해제해 **상주를 약 절반**으로 줄인다.

    되돌릴 수 없으므로 **추론 전용**이다. 학습·진단(gradient 필요)에서는 절대 켜지 않는다.

    `unpack_cache=True` 는 P034 **단계3C** — int8 언팩을 **유니크 모듈당 1회**로 줄인다.
    `int8_store=True` 일 때만 의미가 있고, 결과는 **비트 동일**해야 한다.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(ckpt_path) if ckpt_path else CKPT / f"{arch}.pt"
    st = torch.load(path, map_location=device)
    cfg = TMTConfig(**st["cfg"])
    model = TiedMLPTransformer(cfg).to(device)
    model.load_state_dict(_strip(st["model"]))
    model.set_anneal(1.0)                       # 배포 상태(완전 삼진)에서 추론
    model.eval()
    model.freeze_quant()                        # ★삼진화 1회만(결과 014: 재계산이 CPU 시간의 ~79%)
    if drop_latent:
        model.drop_latent()                     # ★P034 단계2
    if int8_store:
        model.to_int8()                         # ★P034 단계3
    if unpack_cache:
        if not int8_store:
            raise ValueError("unpack_cache 는 int8_store 와 함께만 의미가 있다 "
                             "(fp32 경로에는 언팩이 없다).")
        model.enable_unpack_cache(True)         # ★P034 단계3C
    return model, cfg, device


def _pick(logits, temperature, top_k):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


@torch.no_grad()
def sample(model, cfg, tok, prompt, max_new=100, temperature=0.8, top_k=40, device=None,
           use_cache=True, stop_at_eos=True, eos_id=None):
    """★샘플링 루프의 **단일 소스**. 이미 로드된 모델을 받아 텍스트를 이어쓴다.

    `generate()`(체크포인트 경로로 매번 로드)와 여러 프롬프트를 한 모델로 돌리는 도구
    (`scripts/probe_prompts.py`)가 **같은 코드**를 쓰도록 분리했다.
    P029 최초판이 이 루프를 따로 구현했다가 `cfg.seq_len`(존재하지 않음, 실제는 `max_seq_len`)
    로 크래시했다 — 중복 구현이 원인이었으므로 여기로 합쳤다.

    P030 단계1 추가:
      * `use_cache=True`  — KV 캐시. 매 토큰 전체 재계산(O(T²))을 증분(O(T))으로 바꾼다.
        **`use_cache=False` 는 정확성 대조용 폴백**이다. 그리디(`temperature=0`)로 두
        경로의 출력이 같아야 한다 — 다르면 캐시 구현이 틀린 것이다.
      * `stop_at_eos`     — `<eos>` 에서 멈춘다. 없으면 문서 경계를 넘어 계속 생성한다
        (결과 013 에서 한국어 프롬프트에 영문 문서가 난입한 원인).
    """
    device = device or next(model.parameters()).device
    if eos_id is None:
        eos_id = tok.token_to_id("<eos>")
        if eos_id is None:
            eos_id, stop_at_eos = -1, False
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    dev_type = device if isinstance(device, str) else device.type
    ac = dict(dtype=torch.bfloat16, enabled=(dev_type == "cuda"))

    past = None
    for _ in range(max_new):
        if use_cache:
            # 첫 스텝은 프롬프트 전체(prefill), 이후는 직전 1토큰만 흘린다.
            xin = x if past is None else x[:, -1:]
            past_len = 0 if past is None else next(iter(past.values()))[0].shape[2]
            if past_len + xin.shape[1] > cfg.max_seq_len:
                # 컨텍스트 초과 → 캐시를 버리고 뒤쪽 창으로 재구축(절대위치 RoPE 라 슬라이스 불가)
                past, xin = None, x[:, -(cfg.max_seq_len - 1):]
            with torch.autocast(dev_type, **ac):
                logits, past = model(xin, past_kv=past, use_cache=True)
            logits = logits[:, -1, :].float()
        else:
            with torch.autocast(dev_type, **ac):
                logits = model(x[:, -cfg.max_seq_len:])[:, -1, :].float()
        nxt = _pick(logits, temperature, top_k)
        x = torch.cat([x, nxt], dim=1)
        if stop_at_eos and int(nxt.item()) == eos_id:
            break
    return tok.decode(x[0].tolist())


def generate(prompt, arch="tied", data="ko-en", max_new=100, temperature=0.8,
             top_k=40, ckpt_path=None, device=None, use_cache=True, stop_at_eos=True):
    from tokenizers import Tokenizer
    model, cfg, device = load_model(arch, ckpt_path, device)
    tok = Tokenizer.from_file(str(tokenizer_path(data)))
    text = sample(model, cfg, tok, prompt, max_new=max_new, temperature=temperature,
                  top_k=top_k, device=device, use_cache=use_cache, stop_at_eos=stop_at_eos)
    print(text)
    return text


def check_cache_equivalence(model, cfg, tok, prompt, max_new=24, device=None):
    """★캐시 정확성 검증: 그리디(temperature=0)면 캐시 유/무 출력이 **완전히 같아야** 한다.

    다르면 캐시 구현이 틀린 것이고, 그 상태의 속도 측정은 아무 의미가 없다.
    가장 틀리기 쉬운 두 곳:
      1. RoPE 절대위치 — 캐시 사용 시 `rope[:T]` 가 아니라 `rope[past_len:past_len+T]`
      2. SDPA 마스크 — q_len < kv_len 에서 `is_causal=True` 는 **좌상단 정렬**이라 조용히 틀린다
    """
    a = sample(model, cfg, tok, prompt, max_new=max_new, temperature=0.0,
               device=device, use_cache=True, stop_at_eos=False)
    b = sample(model, cfg, tok, prompt, max_new=max_new, temperature=0.0,
               device=device, use_cache=False, stop_at_eos=False)
    ok = (a == b)
    print(f"  [캐시검증] {'일치 ✅' if ok else '불일치 ❌ — 캐시 구현이 틀렸다'}  {prompt!r}")
    if not ok:
        print(f"    캐시 사용: {a[:160]!r}\n    캐시 미사용: {b[:160]!r}")
    return ok
