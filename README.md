환경 설정 (colab)
---
```
from google.colab import drive
import os

# 1. 드라이브 마운트
drive.mount('/content/drive')

# 2. 프로젝트 경로 변수 설정
project_path = '/content/drive/MyDrive/PokeyProjectGoldVer_hwlee_20251124'
requirements_file = os.path.join(project_path, 'requirements_v3.txt')

# 3. 작업 디렉토리 변경 (이 부분이 가장 중요합니다!)
# !cd 명령어가 아닌 %cd 매직 커맨드를 써야 쉘 환경 전체에 적용됩니다.
%cd "{project_path}"

# 4. 의존성 설치 (이제 경로가 변경되었으므로 requirements.txt만 써도 되지만, 절대경로 유지도 괜찮습니다)
if os.path.exists(requirements_file):
    print(f"Installing packages from {requirements_file}...")
    !pip install -r "{requirements_file}"
else:
    print(f"Error: requirements.txt not found at {requirements_file}")

# 5. 실행
# 이미 %cd로 폴더 안에 들어왔으므로 파일명만 적어도 실행됩니다.
script_name = 'train_transfer_v2.7_colab.py'

if os.path.exists(script_name):
    print(f"Executing {script_name} from {os.getcwd()}...")
    !python "{script_name}"
else:
    print(f"Error: File not found at {script_name}")
```

변경 사항
---
1. Pokemon Red -> Gold
   : map, events 등 변경 完
2. Reinforcement Learning
   1) Transfer
      - Gold 버전의 달라진 환경에서도 Red 처럼 게임을 플레이할 수 있도록 전이학습
      - code: train_transfer_v*.py (최신: train_transfer_v2.3)
   2) Reward System
      - 전이 학습된 모델 성능 향상을 위해 보상체계 수정 계획 중
   3) Model
      - 현재 적용 모델: PPO
      - RPPO 등 성능 향상 가능한 모델 있는지 확인 必

환경설정(Window)
---
1. 가상환경 생성 및 활성화 (Python 3.13)
```powershell
# 1. python 3.10 버전으로 'poke_env'라는 이름의 가상환경 생성
conda create -n poke_env_2 python=3.10 -y

# 2. 생성한 가상환경 활성화
conda activate poke_env_2

# (필요시) 기존 가상환경 비활성화
# deactivate
```

2. `requirements.txt` 파일
```powershell
pip install -r requirements.txt
    
# (필요시) 캐시 제거
pip cache purge
```

```txt
# 오류시
# requirements_v3.txt
pyboy==2.6.0
gymnasium
stable_baselines3
torch
numpy
pandas
scikit-image
matplotlib
mediapy
pillow
einops
wandb
tensorboard
websockets
PySDL2
pysdl2-dll
```

3. init 파일 생성
```powershell
python make_state.py
```
