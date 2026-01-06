# # run_with_metamon.py
# import os
# from pathlib import Path

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.utils import set_random_seed
# from metamon_offline.bc.bc_policy_wrapper import BCPolicyWrapper

# from GoldEnv import GoldEnv
# import numpy as np
# import random

# import torch

# def encode_gold_obs(obs):
#     """
#     BC 모델과 차원 맞추기용 (obs_dim = 7)
#     """

#     # 1. 체력 (1)
#     hp = float(obs["health"][0])

#     # 2. 레벨 요약 (1) - 평균만
#     lvl = float(obs["level"].mean())

#     # 3. 배지 개수 (1)
#     badges = float(obs["badges"].sum())

#     # 4. 이벤트 진행도 (1)
#     events = float(obs["events"].sum())

#     # 5~7. 최근 행동 요약 (3)
#     ra = obs["recent_actions"]
#     ra_mean = float(ra.mean())
#     ra_max  = float(ra.max())
#     ra_last = float(ra[-1])

#     vec = torch.tensor(
#         [hp, lvl, badges, events, ra_mean, ra_max, ra_last],
#         dtype=torch.float32
#     )

#     return vec

# # GoldEnv 기준 action index 직접 반환
# # 0: DOWN, 3: UP, 4: A
# def bc_to_gold_action(bc_action):
#     if bc_action in [0, 1]:
#         return 4   # A
#     elif bc_action == 2:
#         return 3   # UP
#     elif bc_action == 3:
#         return 0   # DOWN


# # ===== A 플랜 액션 제한 =====
# # GoldEnv.valid_actions 인덱스 (GoldEnv.py 기준)
# # 0: DOWN, 3: UP, 4: A  【turn6:11†GoldEnv.py†L30-L38】
# # 수정
# A_PLAN_ACTIONS = [4, 3, 0, 1, 2]  
# # A, UP, DOWN, LEFT, RIGHT  


# def read_agent_enabled(flag_file: str = "agent_enabled.txt") -> bool:
#     """
#     agent_enabled.txt 첫 줄이 'yes'로 시작하면 True.
#     파일 없으면 False.
#     """
#     try:
#         with open(flag_file, "r", encoding="utf-8") as f:
#             line = f.readline().strip().lower()
#         return line.startswith("yes")
#     except FileNotFoundError:
#         return False
#     except Exception as e:
#         print(f"[경고] agent_enabled.txt 읽기 실패: {e}")
#         return False


# def project_to_a_plan(action_idx: int) -> int:
#     """
#     모델이 뽑은 action_idx(0~6)를
#     A플랜 허용 액션(A/UP/DOWN) 중 하나로 "투영"합니다.

#     방식:
#     - 이미 허용이면 그대로
#     - 아니면 간단히 (action_idx % 3)로 매핑해서 A/UP/DOWN 중 하나로 보냄
#       (완전 랜덤처럼 흔들리지 않게, 결정적 매핑)
#     """
#     if action_idx in A_PLAN_ACTIONS:
#         return action_idx
#     return A_PLAN_ACTIONS[action_idx % len(A_PLAN_ACTIONS)]


# def main():
#     set_random_seed(0)

#     # ===== 환경 설정 =====
#     sess_path = Path("session_run_with_a_plan")
#     sess_path.mkdir(exist_ok=True)

#     env_config = {
#         "headless": False,
#         "save_final_state": True,
#         "early_stop": False,
#         "action_freq": 24,
#         "init_state": "./init.state",
#         "max_steps": 2**23,              # 필요하면 줄여도 됨
#         "print_rewards": True,
#         "save_video": False,
#         "fast_video": True,
#         "session_path": sess_path,
#         "gb_path": "./PokeGold.gbc",
#         "debug": False,
#         "sim_frame_dist": 2_000_000.0,
#         "extra_buttons": False,
#     }

#     env = DummyVecEnv([lambda: GoldEnv(env_config)])

#     bc_policy = BCPolicyWrapper(
#         "metamon_offline/bc/data/bc_policy.pt"
#     )


#     # ===== 모델 로드/생성 =====
#     # 1) 학습된 zip 체크포인트가 있으면 그걸 로드해서 쓰는 걸 추천.
#     #    없으면 그냥 새 PPO 모델 만들어서 "랜덤 같은 행동"이 나올 수 있음.
#     checkpoint_zip = os.environ.get("MODEL_ZIP", "").strip()  # 환경변수로도 지정 가능
#     if checkpoint_zip and os.path.exists(checkpoint_zip):
#         print(f"[로드] PPO 체크포인트 사용: {checkpoint_zip}")
#         model = PPO.load(checkpoint_zip, env=env)
#     else:
#         # 기본(무학습) 모델
#         print("[안내] MODEL_ZIP 미지정/없음 → 기본 PPO 모델(무학습)로 실행합니다.")
#         model = PPO("MultiInputPolicy", env, verbose=0)

#         # ===== 실행 루프 =====
#     obs = env.reset()
#     print("\n=== run_with_metamon.py 실행 시작 ===")
#     print(" - agent_enabled.txt가 'yes'면 에이전트가 입력을 보냅니다")
#     print(" - 'no'면 관전(틱만 진행)")
#     print("====================================\n")

#     # ====== 멈춤 방지용 상태 변수 ======
#     last_coord = None
#     same_coord_steps = 0

#     last_action = None
#     same_action_steps = 0

#     # 화면/상태가 “안 변하는 느낌”을 완화하기 위한 카운터
#     stall_steps = 0

#     # stuck 탈출 모드(몇 스텝 동안 강제로 랜덤 이동)
#     escape_steps = 0

#     # 전투에서 텍스트/메뉴로 멈췄을 때 풀기용
#     battle_stall = 0

#     # 월드에서 A(대화/문/간판) 가끔 누르기
#     WORLD_A_PROB = 0.08  # 8% 정도

#     # stuck 판정 기준 (600은 너무 큼: run 단계에서는 작게)
#     STUCK_THRESHOLD = 40

#     # 같은 액션 연타 제한
#     SAME_ACTION_LIMIT = 12

#     # escape 모드 길이
#     ESCAPE_LEN = 25

#     # 월드 이동 후보(상하좌우)
#     MOVE4 = [0, 1, 2, 3]  # DOWN, LEFT, RIGHT, UP
#     LEFT_RIGHT = [1, 2]

#     # “풀기”에 쓰는 버튼들 (A/B/START)
#     BTN_A = 4
#     BTN_B = 5
#     BTN_START = 6

#     while True:
#         agent_enabled = read_agent_enabled("agent_enabled.txt")

#         if agent_enabled:
#             env0 = env.envs[0]

#             # ---- 현재 좌표/맵 기반으로 stuck 감지 ----
#             try:
#                 coord = env0.get_game_coords()  # (x, y, map)
#             except Exception:
#                 coord = None

#             if coord is not None and coord == last_coord:
#                 same_coord_steps += 1
#             else:
#                 same_coord_steps = 0
#                 last_coord = coord

#             # GoldEnv의 기존 stuck 카운트도 함께 사용(둘 중 하나만 걸려도 탈출)
#             stuck_by_seen = False
#             try:
#                 stuck_by_seen = (env0.get_current_coord_count_reward() == 1)
#             except Exception:
#                 stuck_by_seen = False

#             stuck_now = (same_coord_steps >= STUCK_THRESHOLD) or stuck_by_seen

#             # ---- 전투/비전투 분기 ----
#             in_battle = False
#             try:
#                 in_battle = env0.is_in_battle()
#             except Exception:
#                 in_battle = False

#             if in_battle:
#                 # =========================
#                 # 전투 정책: 기본은 A로 진행
#                 # 가끔 B로 텍스트/메뉴 멈춤 풀기
#                 # =========================
#                 battle_stall += 1

#                 # 너무 오래 A만 눌러서 변화가 없으면 B를 한번 섞어줌
#                 # (전투 결과 텍스트, 배틀 종료 직후, '배울래?' 같은 상황에서 도움이 됨)
#                 if battle_stall % 55 == 0:
#                     action_idx = BTN_B
#                 else:
#                     action_idx = BTN_A

#                 # 전투 중에는 escape 모드/월드 로직 끄기
#                 escape_steps = 0
#                 same_action_steps = 0

#             else:
#                 # =========================
#                 # 월드 정책: "멈춤 방지 랜덤 워크"
#                 # - stuck면 ESCAPE 모드로 강제 전환
#                 # - 같은 방향 연타 제한
#                 # - 가끔 A(대화/문/간판)
#                 # =========================
#                 battle_stall = 0  # 전투 끝났으니 초기화

#                 if stuck_now and escape_steps == 0:
#                     escape_steps = ESCAPE_LEN

#                 if escape_steps > 0:
#                     # stuck 탈출: 좌우 우선 + 상하 섞기
#                     # (문에 끼거나 길막에서 빠져나오게 하려고)
#                     action_idx = random.choice(LEFT_RIGHT + MOVE4)
#                     escape_steps -= 1
#                 else:
#                     # 평상시: 상하좌우 + 가끔 A
#                     if random.random() < WORLD_A_PROB:
#                         action_idx = BTN_A
#                     else:
#                         action_idx = random.choice(MOVE4)

#                 # 같은 액션 연타 제한(예: UP만 미친 듯이)
#                 if action_idx == last_action:
#                     same_action_steps += 1
#                 else:
#                     same_action_steps = 0
#                     last_action = action_idx

#                 if same_action_steps >= SAME_ACTION_LIMIT:
#                     # 다른 방향으로 강제 변경
#                     if action_idx in MOVE4:
#                         others = [a for a in MOVE4 if a != action_idx]
#                         action_idx = random.choice(others)
#                     else:
#                         # A를 너무 연타했으면 이동으로 강제
#                         action_idx = random.choice(MOVE4)

#                     same_action_steps = 0
#                     last_action = action_idx

#             # ---- “메뉴/텍스트박스 등으로 멈춘 느낌”을 완화: 주기적으로 START 섞기 ----
#             # (월드에서 START는 메뉴 열기라 호불호 있는데,
#             #  너무 오래 변화 없을 때만 한 번 넣어서 풀리게 하는 용도)
#             if (not in_battle) and (same_coord_steps >= STUCK_THRESHOLD * 2):
#                 # 아주 오래 정체면 START 한번
#                 action_idx = BTN_START
#                 escape_steps = ESCAPE_LEN  # 메뉴 열렸을 수 있으니 탈출 모드로 이어감

#             # ---- 실제 step ----
#             obs, rewards, dones, infos = env.step([action_idx])

#             if bool(dones[0]):
#                 print("[종료] done=True → 에피소드 종료")
#                 break

#         else:
#             # 사람 관전 모드
#             env.envs[0].pyboy.tick(1, True)
#             obs = env.envs[0]._get_obs()

            # ===== 수동 저장용 (임시) =====
            # if os.path.exists("SAVE_INIT_STATE.flag"):
            #     env.envs[0].pyboy.save_state("init.state")
            #     print("✅ init.state 저장 완료")
            #     os.remove("SAVE_INIT_STATE.flag")


#             truncated = env.envs[0].step_count >= env.envs[0].max_steps - 1
#             if truncated:
#                 print("[종료] 최대 스텝 도달 → 에피소드 종료")
#                 break

#     print("실행 종료.")
#     env.close()


# if __name__ == "__main__":
#     main()

# run_with_metamon.py (FIXED)
import os
from pathlib import Path
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from GoldEnv import GoldEnv
from metamon_offline.bc.bc_policy_wrapper import BCPolicyWrapper

from pyboy.utils import WindowEvent

# GoldEnv action index 기준
DOWN, LEFT, RIGHT, UP, A, B, START = 0, 1, 2, 3, 4, 5, 6

EVENT_SEQ = [
    A, A, A,          # 대화 넘기기
    DOWN, A,          # 선택지
    A,
    RIGHT, A,         # 문/출구 시도
    LEFT, A,
    UP, A,
]

def event_action(event_step):
    EVENT_SEQ = [
        4,  # A
        4,  # A
        3,  # UP
        4,  # A
        2,  # RIGHT
        4,  # A
    ]
    return EVENT_SEQ[event_step % len(EVENT_SEQ)]

def read_agent_enabled(flag_file: str = "agent_enabled.txt") -> bool:
    """
    agent_enabled.txt 첫 줄이 'yes'로 시작하면 True.
    파일 없으면 False.
    """
    try:
        with open(flag_file, "r", encoding="utf-8") as f:
            line = f.readline().strip().lower()
        return line.startswith("yes")
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[경고] agent_enabled.txt 읽기 실패: {e}")
        return False


def safe_env_step(vec_env, action_idx: int):
    """
    SB3/Gym 버전에 따라 step 반환이 (obs,reward,done,info) 또는
    (obs,reward,terminated,truncated,info)로 올 수 있어서 둘 다 처리.
    """
    out = vec_env.step([int(action_idx)])
    if len(out) == 4:
        obs, rewards, dones, infos = out
        done = bool(dones[0])
        reward = float(rewards[0])
        return obs, reward, done, infos
    elif len(out) == 5:
        obs, rewards, terms, truncs, infos = out
        done = bool(terms[0] or truncs[0])
        reward = float(rewards[0])
        return obs, reward, done, infos
    else:
        raise RuntimeError(f"Unexpected env.step output len={len(out)}")

from battle_logic import BattleState
# ===== (선택) 전투 BC 입력 인코딩 =====
# def encode_gold_obs(obs):
#     """
#     BC 모델과 차원 맞추기용 (obs_dim=7) — 네 코드 유지.
#     단, 이건 '전투용'으로만 쓰는 걸 권장.
#     """
#     hp = float(obs["health"][0])
#     lvl = float(obs["level"].mean())
#     badges = float(obs["badges"].sum())
#     events = float(obs["events"].sum())

#     ra = obs["recent_actions"]
#     ra_mean = float(ra.mean())
#     ra_max  = float(ra.max())
#     ra_last = float(ra[-1])

#     return torch.tensor([hp, lvl, badges, events, ra_mean, ra_max, ra_last], dtype=torch.float32)

def encode_gold_obs_v11(env_instance):
    """
    BattleState를 사용하여 11차원 벡터를 추출합니다.
    """
    parser = BattleState(env_instance)
    vec = parser.get_battle_vector()
    return torch.tensor(vec, dtype=torch.float32)

def bc_to_gold_action(bc_action: int) -> int:
    """
    네 기존 매핑을 더 안전하게:
    - 예측값이 이상하면 A로 fallback
    """
    bc_action = int(bc_action)
    if bc_action in [0, 1]:
        return A
    if bc_action == 2:
        return UP
    if bc_action == 3:
        return DOWN
    return A


class WorldExplorer:
    """
    월드(필드) 탐험용 규칙 기반 컨트롤러:
    - 같은 방향을 몇 스텝 유지 (지그재그 감소)
    - 좌표가 안 바뀌면(stuck) 즉시 방향 변경
    - 가끔 A(대화/표지판/문 상호작용) 시도
    """
    def __init__(self, hold_min=8, hold_max=20, stuck_limit=25, a_press_prob=0.03):
        self.hold_min = hold_min
        self.hold_max = hold_max
        self.stuck_limit = stuck_limit
        self.a_press_prob = a_press_prob

        self.cur_dir = random.choice([UP, DOWN, LEFT, RIGHT])
        self.hold_left = random.randint(self.hold_min, self.hold_max)

        self.last_coord = None
        self.stuck_count = 0

    def _pick_new_dir(self):
        # 너무 같은 축만 타지 않게 랜덤 + 현재 방향 회피
        dirs = [UP, DOWN, LEFT, RIGHT]
        if self.cur_dir in dirs:
            dirs.remove(self.cur_dir)
        self.cur_dir = random.choice(dirs)
        self.hold_left = random.randint(self.hold_min, self.hold_max)

    def next_action(self, env0):
        # 1) 가끔 A 눌러보기 (NPC/문/표지판)
        if random.random() < self.a_press_prob:
            return A

        # 2) 좌표 기반 stuck 감지
        try:
            x, y, m = env0.get_game_coords()
            coord = (int(x), int(y), int(m))
        except Exception:
            coord = None

        if coord is not None:
            if self.last_coord == coord:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
                self.last_coord = coord

            # stuck면 즉시 방향 변경 + 한 번 A로 "막힘(대화/문)" 탈출 시도
            if self.stuck_count >= self.stuck_limit:
                # 1) A로 한번 비벼보고
                self.stuck_count = 0
                self._pick_new_dir()
                return A

        # 3) 방향 유지(hold) 끝나면 방향 변경
        if self.hold_left <= 0:
            self._pick_new_dir()

        self.hold_left -= 1
        return self.cur_dir


class BattleController:
    """
    전투 상태에서 '가만히' 멈추는 걸 줄이기 위한 간단 매크로:
    - 기본은 A(텍스트/결정)
    - 가끔 DOWN+A로 커맨드/기술 선택 흐름을 밀어줌
    - 아주 가끔 B로 팝업/메뉴 닫기
    """
    def __init__(self):
        self.t = 0

    def next_action(self):
        self.t += 1

        # 아주 가끔 B로 닫기 시도
        if self.t % 90 == 0:
            return B

        # 가끔 DOWN 넣어서 메뉴 선택 변화
        if self.t % 25 == 0:
            return DOWN

        # 대부분 A
        return A
    
# 0xDA22: 파티 내 포켓몬 수 (Pokemon Gold/Silver 기준)
def get_party_count(env):
    return env.read_m(0xDA22)

# 0xD144: 현재 맵 ID (현재 위치 확인용)
def get_current_map(env):
    return env.read_m(0xD144)

def main():
    set_random_seed(0)

    sess_path = Path("session_run_fixed")
    sess_path.mkdir(exist_ok=True)

    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "./init.state",
        "max_steps": 2**23,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "./PokeGold.gbc",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    # env = DummyVecEnv([lambda: GoldEnv(env_config)])

    # # BC policy (전투용으로만 사용 권장)
    # bc_policy = BCPolicyWrapper("metamon_offline/bc/data/bc_policy.pt")

    # # (선택) PPO 체크포인트 있으면 로드. 없으면 그냥 생성(관전/실험용)
    # checkpoint_zip = os.environ.get("MODEL_ZIP", "").strip()
    # if checkpoint_zip and os.path.exists(checkpoint_zip):
    #     print(f"[로드] PPO 체크포인트 사용: {checkpoint_zip}")
    #     model = PPO.load(checkpoint_zip, env=env)
    # else:
    #     print("[안내] MODEL_ZIP 미지정/없음 → PPO(무학습) 객체 생성(실사용은 안 함)")
    #     model = PPO("MultiInputPolicy", env, verbose=0)

    # obs = env.reset()

    # explorer = WorldExplorer(
    #     hold_min=10, hold_max=25,
    #     stuck_limit=25,       # 25번 연속 좌표 고정이면 즉시 탈출
    #     a_press_prob=0.04     # 문/대화 상호작용 조금 더 자주
    # )
    # battler = BattleController()

    # print("\n=== run_with_metamon.py (FIXED) ===")
    # print(" - agent_enabled.txt == yes  → 에이전트 조작")
    # print(" - 그 외               → 관전(틱만 진행)")
    # print(" - 전투: BC/매크로, 월드: 규칙기반 탐험 + stuck 탈출")
    # print("====================================\n")

    # step_i = 0

    # # ===== Mini Dashboard State =====
    # visited_maps = set()

    # last_coord = None
    # coord_stuck_steps = 0

    # battle_steps = 0
    # last_battle_flag = False

    # reward_window = []
    # REWARD_WIN = 200

    # # ===== Event / Stuck tracking =====
    # event_step = 0
    # last_events = 0
    # last_event_step = 0


    # while True:
    #     step_i += 1
    #     reward = 0.0
        
    #     env0 = env.envs[0]
    #     gold_obs = env0._get_obs()

    #     # 좌표 / 맵
    #     try:
    #         x, y, map_id = env0.get_game_coords()
    #         coord = (x, y, map_id)
    #         visited_maps.add(map_id)
    #     except Exception:
    #         coord = None
    #         map_id = -1

    #     # 좌표 stuck 체크
    #     if coord is not None and coord == last_coord:
    #         coord_stuck_steps += 1
    #     else:
    #         coord_stuck_steps = 0
    #         last_coord = coord

    #     # 전투 상태
    #     in_battle = bool(env0.is_in_battle())
    #     if in_battle:
    #         battle_steps += 1
    #     else:
    #         if last_battle_flag and not in_battle:
    #             battle_steps = 0  # 전투 종료 감지
    #     last_battle_flag = in_battle

    #     agent_enabled = read_agent_enabled("agent_enabled.txt")

    #     if not agent_enabled:
    #         # 관전: 에뮬만 tick
    #         env.envs[0].pyboy.tick(1, True)
    #         if env.envs[0].step_count >= env.envs[0].max_steps - 1:
    #             print("[종료] 최대 스텝 도달")
    #             break
    #         continue

    #     env0 = env.envs[0]
    #     in_battle = bool(env0.is_in_battle())

    #     if in_battle:
    #         # 1) 전투: 우선 '안 멈추는' 매크로 기반
    #         # action_idx = battler.next_action()

    #         # 2) (선택) BC도 섞고 싶으면 아래처럼 "가끔만" 사용
    #         gold_obs = env0._get_obs()
    #         bc_input = encode_gold_obs(gold_obs)
    #         if random.random() < 0.30:  # 30%만 BC 사용
    #             bc_action = bc_policy.predict(bc_input)
    #             action_idx = bc_to_gold_action(bc_action)

    #     else:
    #         # 월드: BC 절대 사용 X (여기가 UP만 가는 원흉이었음)
    #         action_idx = explorer.next_action(env0)

    #     env0 = env.envs[0]
    #     gold_obs = env0._get_obs()

    #     events = int(gold_obs["events"].sum())
    #     coord_stuck = coord_stuck_steps  # 네 대시보드에서 계산한 값

    #     event_mode = (
    #         coord_stuck > 200 or
    #         (events == last_events and step_i - last_event_step > 500)
    #     )

    #     if events != last_events:
    #         last_events = events
    #         last_event_step = step_i

    
    #     if event_mode:
    #         action_idx = event_action(event_step)
    #         event_step += 1

    #     obs, reward, done, infos = safe_env_step(env, action_idx)

    #     # 보상 이동 평균
    #     reward_window.append(reward)
    #     if len(reward_window) > REWARD_WIN:
    #         reward_window.pop(0)

    #     if len(reward_window) > REWARD_WIN:
    #         reward_window.pop(0)

    #     avg_reward = sum(reward_window) / len(reward_window)

    #     if done:
    #         print("[종료] done=True")
    #         break

    #     # 디버그 로그(원하면 주석 해제)
    #     # if step_i % 200 == 0:
    #     #     try:
    #     #         x, y, m = env0.get_game_coords()
    #     #         print(f"[{step_i}] battle={in_battle} action={action_idx} coord=({x},{y},{m}) r={reward:.3f}")
    #     #     except Exception:
    #     #         print(f"[{step_i}] battle={in_battle} action={action_idx} r={reward:.3f}")

    #     if step_i % 500 == 0:
    #         events = int(gold_obs["events"].sum())
    #         badges = int(gold_obs["badges"].sum())

    #         print(
    #             "\n========== Mini Dashboard ==========\n"
    #             f"Step            : {step_i}\n"
    #             f"Map ID          : {map_id}\n"
    #             f"Unique Maps     : {len(visited_maps)}\n"
    #             f"Coord           : {coord}\n"
    #             f"Coord Stuck     : {coord_stuck_steps}\n"
    #             f"In Battle       : {in_battle}\n"
    #             f"Battle Steps    : {battle_steps}\n"
    #             f"Events          : {events}\n"
    #             f"Badges          : {badges}\n"
    #             f"Avg Reward({REWARD_WIN}) : {avg_reward:.3f}\n"
    #             "====================================\n"
    #         )

    # print("실행 종료.")
    # env.close()

    # 환경 생성
    dummy_env = GoldEnv(env_config)
    env = DummyVecEnv([lambda: dummy_env])

    # [수정] 모델 경로를 우리가 새로 만들 11차원용으로 지정
    # 만약 아직 학습 전이라면 에러가 날 수 있으니, 
    # 처음엔 데이터 수집용으로 기존 경로를 쓰거나 예외 처리가 필요합니다.
    BC_MODEL_PATH = "metamon_offline/bc/data/bc_policy_v11.pt"
    if os.path.exists(BC_MODEL_PATH):
        bc_policy = BCPolicyWrapper(BC_MODEL_PATH)
        print(f"[알림] 11차원 BC 모델 로드 완료: {BC_MODEL_PATH}")
    else:
        bc_policy = None
        print("[경고] BC 모델이 없습니다! 데이터 수집 모드로 동작합니다.")

    # [추가] 데이터 수집용 버퍼
    collected_data = []

    obs = env.reset()
    explorer = WorldExplorer()
    step_i = 0
    last_saved_state = 0

    try:
        while True:
            step_i += 1
            env0 = env.envs[0]
            agent_enabled = read_agent_enabled("agent_enabled.txt")
            in_battle = bool(env0.is_in_battle())

            # 1) 사람 모드(관전/수동)면: 무조건 tick 해서 화면 갱신
            if not agent_enabled:
                # 1) 화면 갱신
                env0.pyboy.tick(1, True)

                # 2) ===== init.state 수동 저장 트리거 =====
                if os.path.exists("SAVE_INIT_STATE.flag"):
                    with open("init.state", "wb") as f:
                        env0.pyboy.save_state(f)

                    print("✅ init.state 저장 완료")
                    os.remove("SAVE_INIT_STATE.flag")

                continue


            if in_battle:
                # 전투 진입 직후 RAM 안정화용 딜레이 (권장)
                if env0.step_count < 5:
                    action_idx = A
                    obs, reward, done, infos = safe_env_step(env, action_idx)
                    continue

                # 1. 11차원 데이터 추출
                bc_input = encode_gold_obs_v11(env0)
                current_state_np = bc_input.numpy()

                # 2. [핵심 추가] 중복 검사: 이전과 똑같은 화면이면 저장 안 함!
                is_duplicate = False
                if 'last_saved_state' in locals() and last_saved_state is not None:
                    if np.array_equal(current_state_np, last_saved_state):
                        is_duplicate = True

                # 3. 중복이 아닐 때만 데이터 수집
                if not is_duplicate:
                    if bc_policy is not None:
                        bc_action = bc_policy.predict(bc_input)
                    else:
                        bc_action = random.randint(0, 3)

                    collected_data.append({
                        "state": current_state_np,
                        "action": bc_action
                    })
                    last_saved_state = current_state_np # 마지막 상태 기억
                
                # 4. [중요] 멈춤 방지: 5스텝마다 무조건 A를 눌러 대화창을 넘김
                if step_i % 5 == 0:
                    action_idx = A
                else:
                    # 실제 공격 기술 결정 (bc_action이 없을 경우를 대비해 다시 계산하거나 가져옴)
                    if bc_policy is not None:
                        pred_action = bc_policy.predict(bc_input)
                    else:
                        pred_action = random.randint(0, 3)
                    action_idx = bc_to_gold_action(pred_action)
                
            else:
                # 1. 일단 탐험가가 결정한 액션을 가져옴
                action_idx = explorer.next_action(env0)

                # 2. [추가] 월드에서 40번 이상 좌표가 안 바뀌면(Stuck)
                if explorer.stuck_count > 40:
                    # 5번에 한 번씩 B 버튼을 눌러서 지도/메뉴/대화창을 강제로 닫음
                    if explorer.stuck_count % 5 == 0:
                        action_idx = B
                        print("🛑 [월드 Stuck] B 버튼으로 지도/메뉴 탈출 시도!")
                
                # 전투가 끝났으므로 중복 체크용 상태 초기화
                last_saved_state = None

            # 환경 실행
            obs, reward, done, infos = safe_env_step(env, action_idx)

            if step_i % 500 == 0:
                print(f"Step: {step_i} | In Battle: {in_battle} | Data Count: {len(collected_data)}")

            if done: break

    finally:
        # [추가] 종료 시 데이터셋 저장
        if len(collected_data) > 0:
            save_path = "metamon_offline/bc/data/bc_dataset_v11.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True) # 폴더 없으면 생성
            states = torch.tensor([d["state"] for d in collected_data], dtype=torch.float32)
            actions = torch.tensor([d["action"] for d in collected_data], dtype=torch.long)
            torch.save({"states": states, "actions": actions},save_path)
            print(f"\n[완료] {len(collected_data)}개의 전투 데이터가 저장되었습니다.")
        
        env.close()


if __name__ == "__main__":
    main()
