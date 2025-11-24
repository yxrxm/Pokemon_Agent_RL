프로젝트명
---
AI를 포켓몬 마스터로!

한 줄 소개
---
강화학습을 통한 인공지능 모델 개발 프로젝트입니다. 

동기
---
📹https://youtu.be/DcYLT37ImBY?feature=shared
해당 영상을 보고 흥미를 느꼈고 영상 설명란에 깃허브 자료를 보고 비슷하게 개발할 수 있겠다고 생각해서 팀을 구성했고 영상과 같이 클리어를 목표로 더 나은 모델을 개발하는 것을 목표로 프로젝트를 진행하게 되었습니다.

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

환경설정
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
