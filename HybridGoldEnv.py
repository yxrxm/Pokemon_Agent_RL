import numpy as np
import torch
from gymnasium import Env

from GoldEnv import GoldEnv
from metamon_offline.bc.bc_policy_wrapper import BCPolicyWrapper
from battle_logic import BattleState

# GoldEnv action index
DOWN, LEFT, RIGHT, UP, A, B, START = 0, 1, 2, 3, 4, 5, 6


# def encode_battle_obs(obs: dict) -> torch.Tensor:
#     """
#     (지금 가지고 있는 BCPolicyWrapper가 7차원 입력을 기대한다는 전제)
#     전투용 요약 벡터. 기존 run_with_metamon에서 쓰던 방식 재사용.
#     """
#     hp = float(obs["health"][0])
#     lvl = float(np.mean(obs["level"]))
#     badges = float(np.sum(obs["badges"]))
#     events = float(np.sum(obs["events"]))

#     ra = obs["recent_actions"]
#     ra_mean = float(np.mean(ra))
#     ra_max = float(np.max(ra))
#     ra_last = float(ra[-1])

#     return torch.tensor([hp, lvl, badges, events, ra_mean, ra_max, ra_last], dtype=torch.float32)

def encode_battle_obs(obs: dict, env_instance) -> torch.Tensor:
    """
    기존의 7차원 요약을 버리고, BattleState 클래스를 통해 
    정밀한 11차원 전투 벡터를 가져옵니다.
    """
    parser = BattleState(env_instance)
    vec = parser.get_battle_vector()
    return torch.tensor(vec, dtype=torch.float32)

def bc_to_gold_action(bc_action: int) -> int:
    """
    너가 쓰던 매핑 그대로. (필요하면 여기만 바꾸면 됨)
    """
    bc_action = int(bc_action)
    if bc_action in [0, 1]:
        return A
    elif bc_action == 2:
        return UP
    elif bc_action == 3:
        return DOWN
    return A


# class HybridGoldEnv(Env):
#     """
#     GoldEnv를 감싸서:
#     - 전투 중: 오프라인 정책(BC)이 행동 결정
#     - 비전투: 외부(PPO)가 준 행동 그대로 수행
#     """
#     def __init__(self, env_config, bc_ckpt_path: str, bc_prob: float = 1.0):
#         super().__init__()
#         self.env = GoldEnv(env_config)
#         self.bc = BCPolicyWrapper(bc_ckpt_path)
#         self.bc_prob = float(bc_prob)

#         # gym space는 내부 env 것 그대로 노출
#         self.action_space = self.env.action_space
#         self.observation_space = self.env.observation_space

#         # 디버그용 카운터
#         self.last_used_bc = False
#         self.bc_steps = 0
#         self.ppo_steps = 0

#     def reset(self, seed=None, options=None):
#         self.last_used_bc = False
#         self.bc_steps = 0
#         self.ppo_steps = 0
#         return self.env.reset(seed=seed, options=options or {})

#     def step(self, action):
#         # GoldEnv는 action이 "int" 하나를 기대 (Subproc/DummyVecEnv가 list로 감쌈)
#         action = int(action)

#         # 현재 상태 관측(전투 판정용)
#         obs = self.env._get_obs()
#         in_battle = bool(self.env.is_in_battle())

#         # 전투면 BC로 override (확률적으로 섞을 수도 있음)
#         # if in_battle and (np.random.rand() < self.bc_prob):
#         #     bc_in = encode_battle_obs(obs)
#         #     bc_action = self.bc.predict(bc_in)
#         #     action = bc_to_gold_action(bc_action)

#         #     self.last_used_bc = True
#         #     self.bc_steps += 1
#         # else:
#         #     self.last_used_bc = False
#         #     self.ppo_steps += 1

class HybridGoldEnv(Env):
    def __init__(self, env_config, bc_ckpt_path: str, bc_prob: float = 1.0):
        super().__init__()
        self.env = GoldEnv(env_config)
        self.bc = BCPolicyWrapper(bc_ckpt_path)
        self.bc_prob = bc_prob
        self.collected_data = []


    def execute_battle_macro(self, target_move_idx):
        """
        AI가 결정한 기술 번호(0~3)를 받아서 실제 게임 버튼 시퀀스로 실행합니다.
        """
        # 1. '싸우다(Fight)' 메뉴 진입 (A 버튼)
        self.env.run_action_on_emulator(A) 
        
        # 2. 현재 기술 메뉴 커서 위치 확인 ($CC2A)
        # (주의: GoldEnv에 read_m이 구현되어 있어야 함)
        current_cursor = self.env.read_m(0xCC2A) 
        
        # 3. 목표 기술 위치로 이동 (단순화된 로직)
        # 실제로는 상/하/좌/우 복잡하지만, 초기엔 무조건 위(UP)로 끝까지 올린 후 
        # 목표만큼 아래(DOWN)로 내리는 식으로 짜면 확실합니다.
        for _ in range(3): self.env.run_action_on_emulator(UP)
        for _ in range(target_move_idx): self.env.run_action_on_emulator(DOWN)
        
        # 4. 기술 확정 (A 버튼)
        self.env.run_action_on_emulator(A)

    def step(self, action):
        obs = self.env._get_obs()
        in_battle = bool(self.env.is_in_battle())

        if in_battle and (np.random.rand() < self.bc_prob):
            # 1. 11차원 상태 추출
            bc_in = encode_battle_obs(obs, self.env) 
            
            # 2. 모델 예측 (0~3: 기술 번호)
            bc_action = self.bc.predict(bc_in) 
            
            # [추가] 데이터 수집: 현재 상태와 모델이 한 행동을 저장
            # (나중에 이 데이터를 정답지로 재학습할 수 있음)
            self.collected_data.append({
                "state": bc_in.numpy(),
                "action": bc_action
            })

            # 3. 버튼 시퀀스 실행 (매크로)
            self.execute_battle_macro(bc_action)
            
            # 매크로 실행 후의 최종 상태를 위해 가짜 action(A) 리턴
            return self.env.step(A) 

        return self.env.step(action)

    def save_collected_dataset(self, filename="./data/bc_dataset_v11.pt"):
        """수집된 데이터를 .pt 파일로 저장합니다."""
        if not self.collected_data: return
        
        states = torch.tensor([d["state"] for d in self.collected_data], dtype=torch.float32)
        actions = torch.tensor([d["action"] for d in self.collected_data], dtype=torch.long)
        
        torch.save({"states": states, "actions": actions}, filename)
        print(f"--- 데이터셋 저장 완료: {len(states)}건 ---")

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()
