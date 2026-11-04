# spritehatch 작업 이어받기

목표: 유료 PixelLab 없이 로컬에서 PixelLab 수준의 게임 픽셀아트(챔피언 48px, 보스 96px,
아래·왼·오른·위 4방향, 걷기·공격·사망 각 4프레임)를 만드는 것. 나중에 오픈소스로 공개.
비교 기준: `~/Desktop/defense-of-champions/public/assets/champions/champion_ironguard.webp`
(PixelLab 결과, 48px 칸 16열×4행. 행이 방향이 아님: 1행 0~3열 아래 걷기, 2행 0~3열 위 걷기, 3행 0~3열 옆 걷기, 나머지 칸은 공격 등 다른 동작이 이어 붙음).

## 규칙
- 답·커밋 메시지는 한국어, 쉬운 말. 단계마다 원자적 커밋 후 main에 바로 push 허락됨(개인 계정 ymc-crow, 2026-11-02).
- 커밋 전 `.venv/bin/ruff check .` 와 `.venv/bin/python -m pytest -q` (지금 38개 통과).

## 구조
- `cli.py` 정지 그림: Z-Image로 768x320 캔버스에 4방향을 한 장에 그림(치비 OpenPose 뼈대) → `pixelate.downscale_sprites`로 48px.
- `pixelate.py` 줄이기: 공통 색 고르기(`pick_colors`, √개수 가중 k-means) → 중간값 흐림(정지 그림은 스프라이트 한 칸의 3/4 크기, 우리 캔버스에선 5px, 2026-11-04 3→5. 다시 그리기는 3px) → 칸마다 다수결(어두운 색 가중) → 외톨이 점 제거 → 같은 재질 점점이 음영 평평하게(`_flatten`, 이웃 8칸 중 4칸 이상이 가까운 색이면 따라감) → 1px 외곽선. 갇힌 배경(`open_holes`), 떨어진 점(`main_body`) 처리.
  `open_holes`는 가장자리 밝기 중간값이 170 미만일 때만 뚫음(반사광은 밝은 금속 180~206에 둘러싸이고, 지팡이 틈은 87~156). 2026-11-03 기사 투구 반사광이 뚫리던 것 해결.
- `moves.py`/`rig.py` 관절 동작(뼈대 리그). `effects.py` 칼·마법·RotSprite 회전.
- `redraw.py` `--redraw`: 리그 프레임 4장을 2x2 격자로 AI가 다시 그림(img2img + OpenPose), 정지 그림 색만 사용. 쓰러진 프레임도 뼈대를 함께 눕힘. AI에는 몸만 보내고(`render(effects=...)`) 칼·궤적·마법은 다시 그린 뒤 `with_effects`로 얹음. 모든 동작에서 몸이 서 있으면 목 위를 리그 머리로 덮음(`_keep_head`, AI가 얼굴을 지워서). 프롬프트는 '빈손'(칼을 말하면 AI가 칼을 하나 더 그림). 누운 프레임은 리그 그대로. 다시 그린 몸 안 구멍은 리그가 꽉 찬 자리면 리그 색으로 메움(`_patch_holes`).
- 외곽선: 저장 직전 `pixelate.selout`이 테두리를 안쪽 색의 어두운 그늘(밝기 56 이하)로 칠함(PixelLab 방식). 반투명(칼 궤적)은 제외. 다시 그리기 색 목록(`sprite_colors`)은 테두리를 뺀 안쪽 색만.
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
  2026-11-03 정지 그림에 0.6·1.0 재시험: 원본은 옆 투구 모양·음영이 좋아지지만 더 잘게 그려서 48px로 줄이면 얼굴(눈)이 사라짐. 근본 원인은 모델이 48px 크기로 그리는 법을 안 배운 것.
- 팔레트 대비·채도 올리기: 효과 미미(어두움은 모델이 그린 탓).
- denoise 0.45/0.35(옛 리그): 얼룩 거의 그대로, 옆 다리만 뻣뻣해짐. 지금은 앞·뒤 0.6, 옆모습만 0.5(`SIDE_DENOISE`, 0.6이면 다리가 검게 칠해짐). 0.4는 무늬 지저분.
- 다리 어두움은 줄이기 단계(dark_weight) 탓이 아님: AI 원본 격자부터 어두움.
- 격자 칸 288: 앞모습 눈이 사라짐. 384 유지.
- 색 수 16/12: 눈은 살지만 가슴 문양이 뭉개짐. 32 + `_flatten`이 나음.
- 사망: AI에게 기울이기 전 선 자세를 그리게 한 뒤 RotSprite로 기울이기 → 앞모습 얼굴 사라지고 누운 옆모습 디테일 뭉개짐. 리그 그림 유지가 나음.
- 다시 그리기 프롬프트에 'face clearly visible' 추가: 거의 차이 없음(공격 앞모습 면갑 그대로).
- 갇힌 흰 부분을 '어두운 외곽선 비율'로 틈 판정: 마법사 지팡이 틈은 외곽선이 흐려 23~30%라 반사광과 구분 안 됨. 가장자리 밝기로 바꿈.
- 앞·뒤 걷기에서 골반·몸통 몇 도 회전: 48px 몸에 구멍. 다리 통째 들기: 치마 뒤로 숨음.

## 남은 일 (우선순위)
1. (완료 2026-11-02) README 예시 교체. 코드가 크게 바뀌면 아래로 다시 만들 것.
   `spritehatch "cute little knight, shiny bright silver armor, blue cape, red plume on the helmet" -o OUT --seed 7`
   → `spritehatch-animate OUT` → `spritehatch-animate OUT --motions walk,attack,death --redraw` → `python -m spritehatch.showcase OUT/anim`
   (설명에 무기를 넣으면 공격 때 칼이 두 개가 됨 — 넣지 말 것)
2. 사망 누운 프레임: 회전 중심을 몸 가운데로 바꿔 옆모습 머리 잘림 해결, 다시 그리기에서 누운 프레임(가로>1.4×세로)은 리그 그림 유지(`redraw._lying`). 쓰러지는 도중(30~65°)은 웅크린 자세 대신 곧게 편 몸을 기울임(조각남 해결). 사망도 몸이 서 있는 동안은 리그 머리 유지.
3. 반짝이는 갑옷 얼룩: `_flatten`으로 일부 해결(4방향에서 약 55픽셀 정리). 2026-11-04 `_merge_rare` 추가: 스프라이트 안에서
   2픽셀 이하로 쓰인 색을 이웃의 같은 재질 색으로 합침(PixelLab은 이런 색 1~2개, 우리 기사 옆 11→6). 5캐릭터 모두 눈·수정 그대로.
   PixelLab과 비교(2026-11-04): 이웃 대비 평균은 비슷(약 45). PixelLab은 가까운 색을 섞어 결이 있고, 우리는 넓은 면에 검고 흰 점이 튐.
   흐림 5로 일부 해결. 다시 그리기에도 5를 쓸지는 AI로 시험해 봐야 함.
   재질별 음영 단계 수 제한은 버림: PixelLab도 주 재질(갑옷)에 18~21단계를 씀(우리 회색 12단계). 차이는 개수가 아니라 배치(띠 모양 vs 점점이).
4. 앞·뒤 걷기: 다리를 들어 올리던 걸 `squash`(다리를 엉덩이 쪽으로 줄임)로 바꿔 두 다리가 갈려 보임. 망토 자락은 `rig.LIMB_REACH=1.5`(팔·다리 굵기 1.5배 밖은 몸통)로 몸에 붙임. 지팡이 든 마법사로 확인해도 지팡이 안 쪼개짐. 옆모습 가슴 해골 무늬는 반사광 구멍 수정(2026-11-03) 후 사라짐.
5. (해결) 효과 칼: 선으로 그리던 걸 손으로 찍은 칼 그림을 RotSprite로 돌리게 바꿈(`effects._sword_sprite`).
6. 근본 해결 후보(2026-11-03 조사): 48px 원본 크기 스프라이트로 우리 LoRA 학습. 맥은 공식 mflux(`uv tool install --python 3.13 mflux==0.23.0 --with 'pillow<13'`)의 `mflux-train`이 Z-Image 학습 지원(quantize 8 + gradient_checkpointing + low_ram). 단 모델 약 31GB를 따로 받아야 해서 사용자 회사 업무 때문에 보류. 시험 설정은 기사 등 20장×512px. 데이터는 공개 라이선스(CC0) 스프라이트 필요.
   2026-11-04 본 학습 결과(공개 스프라이트 43명→86장, 4bit 모델, 10회 반복, 맥에서 약 5시간, 덮개 닫히면 잠들어 멈춤 → `caffeinate`):
   강도 1.0은 원본이 굵은 픽셀로 나오고 옆 투구·깃털이 좋아지지만 정면 얼굴이 눈 없는 덩어리/빨간 눈 가면이 됨.
   3회째 학습본 강도 0.5가 가장 균형(눈 살아 있고 옆 투구 깃털 좋음). 기사 한 장만 본 것이라 다른 4캐릭터로 확인 전엔 기본값으로 넣지 말 것.
   학습 데이터·설정·스크립트·비교 그림은 `~/Desktop/spritehatch-train`(레포 밖), 학습본은 `~/ComfyUI/models/loras/spritehatch_e258.safetensors`(3회), `_e860`(10회).
   2026-11-04 4캐릭터(마법사·궁수·도적·바바리안) 정지 그림 비교: 깨짐 없음, 마법사·바바리안은 눈·버클 또렷, 궁수·도적은 비슷.
   → `spritehatch ... --style-lora spritehatch_e258.safetensors`(강도 기본 0.5)로 넣음. 학습본을 레포에 안 올려서 기본값은 꺼둠.
   비교 그림 `~/Desktop/spritehatch-train/exp/chars/cmp.png`, 스크립트 `exp/chars_try.py`.
   다시 그리기에도 `spritehatch-animate --style-lora`로 걸 수 있음. 2026-11-04 바바리안 걷기 비교: 차이 작음
   (뒷모습 머리끝이 덜 지저분, 앞모습 두 다리 갈림은 조금 약해짐). 다시 그리기엔 굳이 안 걸어도 됨.
   비교 그림 `~/Desktop/spritehatch-train/exp/redraw/cmp.png`.
   다음: 얼굴 또렷한 공개 스프라이트 더 모아 다시 학습. 2026-11-04 조사: 학습 데이터 43명 중 eldiran 6명이 '얼굴 없음' 스타일
   → 강도 1.0에서 눈 없는 얼굴이 나온 원인 후보. 다시 학습 때 빼고, 후보 2명(`~/Desktop/spritehatch-train/new_raw/후보.md`) 추가.
   학습은 약 5시간·컴퓨터 느려짐 → 사용자가 안 쓸 때(밤) `caffeinate`로 돌릴 것.
7. 몬스터·용 등 사람형이 아닌 것: Quaternius CC0 3D 팩 + Blender 렌더 → 깊이/선화 ControlNet(Z-Image Fun Union이 depth/canny 지원) 방향으로 조사만 해 둠.

## 여러 캐릭터 시험 (기사만 보고 다듬지 말 것)
개선 후 기사 말고도 아래 캐릭터로 정지 그림 + 동작을 만들어 깨지는 데가 없는지 볼 것
(설명에 칼 같은 무기는 넣지 말 것, 공격 때 칼이 두 개 됨):
- `cute little wizard, purple robe, pointy hat, long wooden staff with a glowing blue crystal` --seed 3
- `cute little archer girl, green hood, brown leather vest, short brown boots` --seed 5
- `cute little rogue, black hooded cloak, red scarf over the mouth` --seed 11
- `cute little barbarian, orange braided hair, fur shoulder pads, bare arms` --seed 2

캐릭터 시험 결과(2026-11-02): 궁수·바바리안·마법사 무난. 도적은 검은 망토가 외곽선과 뭉개져 `lift_darks`로 해결.
남은 것: 긴 망토가 다리를 덮는 옆모습 걷기(도적)는 망토 아랫단이 다리 따라 들쭉날쭉.
  2026-11-04 조사: 망토 자락을 '몸에 붙일 픽셀'로 가려낼 규칙 못 찾음. 옆모습 엉덩이 아래 폭(도적 18~19, 기사 망토 포함 16~18)도,
  다리 픽셀이 몸통과 같은 색인 비율(공통 팔레트라 모두 0.8~1.0)도 도적과 기사를 못 가름. 옆모습 두 다리 사이 틈도 5캐릭터 모두 거의 없어
  (0~1줄) 판정 못 함. 규칙으로 가르기보다 AI 다시 그리기나 학습 데이터 쪽에서 풀 문제로 보임.

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
