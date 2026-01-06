# Pokémon Gold 강화학습 프로젝트 (yurim 브랜치)

환경 정의, 실행 스크립트, 학습 스크립트 등 **핵심 코드만 포함**하여 추가함

---

**월드 이동(탐험/전략)** 과 **전투 행동**을 분리하여 설계하였다.

- **월드 이동 및 전략적 판단** → PPO가 학습
- **전투 행동(기술 선택 등)** → 사전 학습된 BC(Behavior Cloning) 정책 사용

---

## 전체 실행 파이프라인 

[실행 / 환경 구동]
run_with_metamon.py
   - PPO 또는 규칙 기반으로 월드 이동
   - 전투는 BC 정책으로 처리
   - 상태(state) / 행동(action) / 보상(reward) 흐름 생성

[학습]
train_gold.py        (메인 PPO 학습)
또는
train_world_ppo.py  (월드 이동 중심 PPO 학습)

   - PPO policy (.zip / .pth) 생성

[검증 / 재실행]
run_with_metamon.py
   - 학습된 PPO 정책을 로드하여 실행
