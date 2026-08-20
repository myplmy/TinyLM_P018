# MobileLLM-125M parity reference provenance

- 기록일: 2026-08-21
- 생성 스크립트: `test_mobile_make_reference.py`
- upstream submodule commit: `6cb80c4064183622a64105c875cdd8a35fc8c56b`
- upstream config: `MobileLLM/configs/125M/config.json`
- 생성 artifact 시각: 2026-08-09 09:07:58 (filesystem timestamp)

## SHA-256

| 파일 | bytes | SHA-256 |
|---|---:|---|
| `mobilellm_125m_state_dict.pt` | 498,656,107 | `1dae1be8be444ffdd7ad79b183d7f5f0f2c9744341a0452426c8b45b8d07a2f8` |
| `reference_inventory.json` | 40,552 | `0589757fce196c6fa4871de3f33363b39462487556090e321fd4a848c7336cd4` |
| `reference_summary.json` | 445 | `36f3bd439eb23b7fd4e08f97b7d09fc2d259eeb8b7777bb026c236c9118ba188` |
| `test_mobile_make_reference.py` | 7,540 | `74e336c98291ae27d82984d492d40df3ab21fa54796878bbbebca57ec581d0b5` |
| `test_mobile_weight_parity.py` | 70,239 | `1406769932d8344b202c5ab25ea3de9957eb1191a11ddc5f9bbe8dcd971bfd1b` |
| upstream `config.json` | - | `95a6e60392bc3c1c0ccc1890f1e53b7fe9685b14fe0d64904bf4541474aa047f` |

## 증거 경계

`test_mobile_make_reference.py`는 공식 source에서 random-initialized dense reference를 생성해 이 디렉터리에 저장한다. 반면 현재 `test_mobile_weight_parity.py`는 `mobilellm_125m_state_dict.pt`를 읽지 않고 같은 프로세스에서 공식 모델을 생성한 뒤 TinyLM에 직접 weight를 복사한다.

따라서 2026-08-09 parity 로그는 공식 source-to-TinyLM parity 증거이지만, 이 `.pt` artifact의 load/round-trip parity 증거는 아니다. 그 gate는 별도 사용자 실행이 필요하다. `.pt`는 root `.gitignore`의 `*.pt` 규칙으로 Git에 포함되지 않는다.
