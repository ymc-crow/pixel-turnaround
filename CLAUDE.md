# spritehatch 작업 이어받기

목표: 유료 PixelLab 없이 로컬에서 PixelLab 수준의 게임 픽셀아트(챔피언 48px, 보스 96px,
아래·왼·오른·위 4방향, 걷기·공격·사망 각 4프레임)를 만드는 것. 나중에 오픈소스로 공개.
비교 기준: `~/Desktop/defense-of-champions/public/assets/champions/champion_ironguard.webp`
(PixelLab 결과, 48px 칸 16열×4행, 행 = 아래·왼·오른·위, 0~3열이 걷기).

## 규칙
- 답·커밋 메시지는 한국어, 쉬운 말. 단계마다 원자적 커밋 후 main에 바로 push 허락됨(개인 계정 ymc-crow, 2026-11-02).
- 커밋 전 `.venv/bin/ruff check .` 와 `.venv/bin/python -m pytest -q` (지금 28개 통과).

## 구조
- `cli.py` 정지 그림: Z-Image로 768x320 캔버스에 4방향을 한 장에 그림(치비 OpenPose 뼈대) → `pixelate.downscale_sprites`로 48px.
- `pixelate.py` 줄이기: 공통 색 고르기(`pick_colors`, √개수 가중 k-means) → 3px 중간값 흐림 → 칸마다 다수결(어두운 색 가중) → 외톨이 점 제거 → 같은 재질 점점이 음영 평평하게(`_flatten`, 이웃 8칸 중 4칸 이상이 가까운 색이면 따라감) → 1px 외곽선. 갇힌 배경(`open_holes`), 떨어진 점(`main_body`) 처리.
- `moves.py`/`rig.py` 관절 동작(뼈대 리그). `effects.py` 칼·마법·RotSprite 회전.
- `redraw.py` `--redraw`: 리그 프레임 4장을 2x2 격자로 AI가 다시 그림(img2img + OpenPose), 정지 그림 색만 사용. 쓰러진 프레임도 뼈대를 함께 눕힘. 걷기·대기는 목 위를 리그 머리로 덮음(`_keep_head`, AI가 얼굴을 지워서). 누운 프레임은 리그 그대로.
- `showcase.py` 모든 동작을 GIF 한 장으로.

## 환경 (중요)
- ComfyUI: `~/ComfyUI`, 반드시 `python main.py --listen 127.0.0.1 --port 8188 --gpu-only` 로 켤 것 (`/comfyui` 스킬이 그렇게 함).
  `--gpu-only` 없으면 모델을 두 벌 들고 27GB를 써서 스왑 → 2분 일이 15분 넘어 시간 초과.
- 맥 32GB에서 Z-Image(UNET 12GB + 텍스트 인코더 8GB)는 빠듯함. 다시 그리기 격자는 `GRID_MAX=768` 넘으면 메모리 넘침(1008/1152에서 실패 확인).
- 첫 실행 때 모델 올리는 동안 ComfyUI가 HTTP 응답을 30초 넘게 못 함 → `comfy.run`이 확인 요청 실패를 무시하고 계속 기다리게 고침.
- ComfyUI가 멈추면 `/interrupt`로 안 풀림(모델 로딩 중). 프로세스를 끄고 wired 메모리가 내려갈 때까지 기다린 뒤 다시 켤 것.
- fp8 저장(`--fp8_e4m3fn-*`)은 MPS에서 LoRA 적용 시 오류. 쓰지 말 것.
- 디스크 여유 약 16GB. 모델 더 받을 때 주의.
- 속도(M1 Pro, `--gpu-only`): 정지 그림 약 1분, 다시 그리기 방향당 약 2분(6단계, 2026-11-02 8→6). `--steps 4`면 75초지만 작은 문양 뭉개짐. 시험은 바뀐 방향·동작 하나만 돌릴 것.

## 해 보고 버린 것 (다시 하지 말 것)
- SDXL 기본: Z-Image보다 얼굴·일관성 나쁨(`--model sdxl`로 남겨둠).
- 1536 캔버스: 48px로 줄이면 지저분. 768이 깔끔하고 3배 빠름.
- 픽셀아트 LoRA(tarn59): 없는 쪽이 설명을 더 잘 따름(두건·불꽃 지팡이). 메모리도 더 씀.
- 팔레트 대비·채도 올리기: 효과 미미(어두움은 모델이 그린 탓).
- denoise 0.45/0.35(옛 리그): 얼룩 거의 그대로, 옆 다리만 뻣뻣해짐. 지금은 앞·뒤 0.6, 옆모습만 0.5(`SIDE_DENOISE`, 0.6이면 다리가 검게 칠해짐). 0.4는 무늬 지저분.
- 다리 어두움은 줄이기 단계(dark_weight) 탓이 아님: AI 원본 격자부터 어두움.
- 격자 칸 288: 앞모습 눈이 사라짐. 384 유지.
- 색 수 16/12: 눈은 살지만 가슴 문양이 뭉개짐. 32 + `_flatten`이 나음.
- 사망: AI에게 기울이기 전 선 자세를 그리게 한 뒤 RotSprite로 기울이기 → 앞모습 얼굴 사라지고 누운 옆모습 디테일 뭉개짐. 리그 그림 유지가 나음.
- 다시 그리기 프롬프트에 'face clearly visible' 추가: 거의 차이 없음(공격 앞모습 면갑 그대로).
- 앞·뒤 걷기에서 골반·몸통 몇 도 회전: 48px 몸에 구멍. 다리 통째 들기: 치마 뒤로 숨음.

## 남은 일 (우선순위)
1. (완료 2026-11-02) README 예시 교체. 코드가 크게 바뀌면 아래로 다시 만들 것.
   `spritehatch "cute little knight, shiny bright silver armor, blue cape, red plume on the helmet" -o OUT --seed 7`
   → `spritehatch-animate OUT` → `spritehatch-animate OUT --motions walk,attack,death --redraw` → `python -m spritehatch.showcase OUT/anim`
   (설명에 무기를 넣으면 공격 때 칼이 두 개가 됨 — 넣지 말 것)
2. 사망 누운 프레임: 회전 중심을 몸 가운데로 바꿔 옆모습 머리 잘림 해결, 다시 그리기에서 누운 프레임(가로>1.4×세로)은 리그 그림 유지(`redraw._lying`). 남은 것: 쓰러지는 도중(30~65°) 프레임이 거침.
3. 반짝이는 갑옷 얼룩: `_flatten`으로 일부 해결(4방향에서 약 55픽셀 정리). 더 하려면 재질별 음영 단계 수 제한 검토.
4. 앞·뒤 걷기: 다리를 들어 올리던 걸 `squash`(다리를 엉덩이 쪽으로 줄임)로 바꿔 두 다리가 갈려 보임. 망토 자락은 `rig.LIMB_REACH=1.5`(팔·다리 굵기 1.5배 밖은 몸통)로 몸에 붙임. 지팡이 든 마법사로 확인해도 지팡이 안 쪼개짐. 남은 것: 옆모습 다시 그리기에서 가슴 무늬가 해골처럼 변함.
5. 몬스터·용 등 사람형이 아닌 것: Quaternius CC0 3D 팩 + Blender 렌더 → 깊이/선화 ControlNet(Z-Image Fun Union이 depth/canny 지원) 방향으로 조사만 해 둠.

## 비교하는 법
같은 배율로 나란히 놓고 본다:
```python
from PIL import Image
import spritehatch.pixelate as px
pl = Image.open('.../champion_ironguard.webp').convert('RGBA')
ours = Image.open('OUT/anim/walk.png')
row = [pl.crop((c*48, 0, c*48+48, 48)) for c in range(4)] + [ours.crop((c*48, 0, c*48+48, 48)) for c in range(4)]
px.preview(px.sheet(row), 4).save('cmp.png')
```
