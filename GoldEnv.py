#GoldEnv 보상 체계 파일
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from collections import deque # 최근 방문 좌표를 저장하기 위해 필요**AI 무기력 문제 발생**
import os
import utils
import ai_coach



class GoldEnv(gym.Env):
    def __init__(self, config):
        super(GoldEnv, self).__init__()

        #기본 설정값
        self.config = config
        self.save_video = config.get("save_video", False)
        self.headless = config.get("headless", True)
        self.max_steps = config.get("max_steps", 2048 * 10)
        self.gb_path = config["gb_path"]
        self.init_state = config.get("init_state", "game_file/init.state")
        self.action_freq = config.get("action_freq", 24)

        #내부 변수 -> max_step이 되면 초기화하고 다시 실행함.
        self.step_count = 0 #AI가 걸은 횟수
        self.total_reward = 0 #총 보상의 수
        self.current_badge_count = 0 #게임 내 뱃지의 개수
        
        #이전 상태 저장용 (변화 감지)
        self.prev_money = 0
        self.prev_badges = 0
        self.prev_exp = 0  # 경험치
        self.prev_level_sum = 0  # 레벨 합계
        self.prev_hp = 0  # 체력
        self.prev_max_hp = 0  # 최대 체력
        self.prev_battle_type = 0  # 전투 상태
        self.prev_enemy_hp = 0  # 적 체력

        #통계용 변수
        self.seen_coords = set() #게임 내 탐험 좌표
        self.max_level_sum = 0 #게임에서 만난 최대 레벨
        self.death_count = 0
        self.heal_battle_count = 0  # 전투 중 회복 (물약/기술)
        self.heal_field_count = 0  # 필드 중 회복 (포켓몬센터/물약)
        
        #추가 변수들
        self.steps_on_map = 0 #현재 맵에서 보낸 시간
        self.prev_map_id = -1 #이전 맵 ID (맵 변경 감지용)
        self.coord_history = deque(maxlen=100) #최근 100스텝 좌표 저장 // **AI 무기력 문제 발생**
        
        # 추가
        self.handling_battle = False
        # __init__ 맨 아래쯤에 추가
        self.battle_skip_steps = 0
        self.battle_skip_last_type = 0
        self.mode = config.get("mode", "train")
        self.is_train = (self.mode == "train")
        self.is_play = (self.mode == "play")
        # 전투 타임아웃 카운터
        self.battle_steps = 0
        self.max_battle_steps = 120 # 앙이   # 약 몇 초 (조절 가능)

        self.stuck_steps = 0
        self.max_stuck_steps = 200  # 조절 가능

        self.prev_coord = None

        self.visited_maps = set()
        

        #LLM AI 설정값
        ai_conf = config.get("ai_config", None)
        if ai_conf and ai_conf.get("use_ai_coach", False):
            self.coach = ai_coach.LLMCoach(ai_conf)
            self.coach_interval = ai_conf["coach_interval"]
        else:
            self.coach = None
            self.coach_interval = 9999999

        #PyBoy 에뮬레이터 실행
        window_type = "null" if self.headless else "SDL2"
        #SDL2: Simple DirectionMedia Layer 2의 약자 실제 모니터 윈도우 창으로 띄워주는 역할
        self.pyboy = PyBoy(self.gb_path, window=window_type)

        #init.state 시작 파일 불러오기
        if self.init_state and os.path.exists(self.init_state):
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

        #에이전트가 누를 때 사용하는 버튼
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B #WindowEvent.PRESS_BUTTON_START,
            #WindowEvent.PRESS_BUTTON_SELECT
        ]

        #에이전트가 뗄 때 사용하는 버튼
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B #WindowEvent.RELEASE_BUTTON_START,
            #WindowEvent.RELEASE_BUTTON_SELECT
        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        #관찰 공간 // 우리가 화면을 보듯이 AI가 보는 화면 규격 // PPO 정책에 사용함 obs(현재 화면)에 따라 미래 보상을 예측하여 행동함.
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 144, 160), dtype=np.uint8)

    #AI와 commands.txt를 통해서 대화를 가능하게 함.
    def check_command_file(self):
        """
        commands.txt 파일을 읽어서 명령이 있으면 수행하고 내용을 지움
        """
        cmd_file = "commands.txt"

        #파일 존재 여부 확인
        if not os.path.exists(cmd_file):
            return

        try:
            with open(cmd_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

            #빈 내용은 제외
            if not content:
                return

            print(f"명령 확인: {content}")

            if content == "조언":
                if self.coach:
                    obs = self.GetObs()
                    advice = self.coach.ask_advice(obs, "사용자가 직접 조언을 요청했어.")
                    print(f" {advice}")
                else:
                    print("코치 없음.")

            #명령어 수행 후, 파일을 비우기
            with open(cmd_file, "w", encoding="utf-8") as f:
                f.write("")

        #파일이 lock인 경우.
        except Exception as e:
            print(f"에러 발생 : {e}")
            pass
        
    # AI의 Step과 Step당의 Update 사항
    def step(self, action):
        self.step_count += 1

        # 무조건 obs를 먼저 만들어 둔다 (보험)
        obs = self.GetObs()

        # -----------------------------------------------------------
        # 1. 사용자 개입 모드 확인 (agent_enabled.txt)
        # -----------------------------------------------------------
        is_user_control = False
        check_file = "agent_enabled.txt"
        if os.path.exists(check_file):
            try:
                with open(check_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content == "yes":
                    is_user_control = True
            except Exception:
                pass

        # -----------------------------------------------------------
        # 2. 모드에 따른 행동 처리 (✅ 최종 안정화 버전)
        # -----------------------------------------------------------
        # battle_type = self.pyboy.memory[utils.MEM_BATTLE_TYPE]
        # battle_now = (battle_type != 0)
        # current_tick = 24 if battle_now else self.action_freq

        # if self.config.get("instant_win_battles", False) and battle_now:
        #     enemy_max_hp = utils.read_be16(self.pyboy, utils.MEM_ENEMY_MAX_HP)
        #     my_max_hp = utils.read_be16(self.pyboy, utils.MEM_BATTLE_HP_MAX)
        #     enemy_hp = utils.read_be16(self.pyboy, utils.MEM_ENEMY_HP)

        #     self.pyboy.tick(current_tick)
        #     # 1️⃣ 등장 연출 / 로딩
        #     if enemy_max_hp == 0 or my_max_hp == 0:
        #         print(1)
        #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        #         self.pyboy.tick(current_tick)
        #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

        #     # 2️⃣ 실제 전투 단계 → 여기서만 즉살
        #     elif enemy_hp > 0:
        #         print(2)
        #         # 기술 사용 직후에만 HP 0
        #         self.pyboy.memory[utils.MEM_ENEMY_HP] = 0
        #         self.pyboy.memory[utils.MEM_ENEMY_HP + 1] = 0

        #         # 싸운다 → 기술 선택 → 데미지
        #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        #         self.pyboy.tick(current_tick)
        #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

        #     # 3️⃣ 승리 / 경험치 / 종료
        #     else:
        #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
        #         self.pyboy.tick(current_tick)
        #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)

        # else:
        #     self.AI_action(action, current_tick)

        # -----------------------------------------------------------
        # 2. 모드에 따른 행동 처리 (✅ 분기 정리)
        # -----------------------------------------------------------
        battle_type = self.pyboy.memory[utils.MEM_BATTLE_TYPE]
        battle_now = (battle_type != 0)

        # 전투는 작은 tick, 필드는 action_freq
        battle_tick = 2
        field_tick = self.action_freq

        # 2-1) 사용자 조작이면: AI 입력 금지, 그냥 tick만
        if is_user_control:
            self.pyboy.tick(battle_tick if battle_now else field_tick)

        # 2-2) 학습 모드 + instant win + 전투면: 스킵 로직
        elif self.is_train and self.config.get("instant_win_battles", False) and battle_now:
            self.battle_skip_step()

        # 2-3) 그 외는 정상 AI action
        else:
            self.AI_action(action, battle_tick if battle_now else field_tick)

        # -----------------------------------------------------------
        # 2-4) 전투 타임아웃 관리 (핵심)
        # -----------------------------------------------------------
        if battle_now:
            self.battle_steps += 1
        else:
            self.battle_steps = 0

        # 전투가 너무 오래 지속되면 에피소드 종료
        if self.battle_steps > self.max_battle_steps:
            truncated = True
            info = {
                "debug/battle_timeout": 1
            }

            return obs, 0.0, False, truncated, info


        # -----------------------------------------------------------
        # 3. 데이터 업데이트 및 보상 계산 (기존 로직)
        # -----------------------------------------------------------

        # 현재 뱃지 수 확인
        self.current_badge_count = utils.get_badges(self.pyboy)

        # 맵 정보 읽기 및 체류 시간 계산
        cur_map_grp = utils.read_uint8(self.pyboy, utils.MEM_MAP_GROUP)
        cur_map_num = utils.read_uint8(self.pyboy, utils.MEM_MAP_NUMBER)
        cur_map_id = (cur_map_grp << 8) | cur_map_num

        cur_x = utils.read_uint8(self.pyboy, utils.MEM_X_POS)
        cur_y = utils.read_uint8(self.pyboy, utils.MEM_Y_POS)

        self.coord_history.append((cur_x, cur_y, cur_map_id))
        
        # 추가
        # 현재 좌표
        current_coord = (cur_map_id, cur_x, cur_y)
        
        # stuck 감지
        if self.prev_coord is None:
            self.prev_coord = current_coord
            self.stuck_steps = 0
        elif current_coord == self.prev_coord:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
            self.prev_coord = current_coord


        # 신규 좌표인지 판별 (아직 add 하지 마!)
        is_new_coord = current_coord not in self.seen_coords

        # GetReward에서 쓰기 위해 클래스 변수로 저장
        self.is_new_coord = is_new_coord

        # 이제서야 기록
        self.seen_coords.add(current_coord)

        if cur_map_id == self.prev_map_id:
            self.steps_on_map += 1
        else:
            self.steps_on_map = 0
            self.prev_map_id = cur_map_id

        # 레벨 합계 계산
        cur_level_sum = utils.get_level_sum(self.pyboy)
        if cur_level_sum > self.max_level_sum:
            self.max_level_sum = cur_level_sum

        # 필드 stuck too long → episode cut (여기!)
        if self.stuck_steps > self.max_stuck_steps:
            truncated = True
            info = {
                "debug/stuck_timeout": 1
            }
            return obs, 0.0, False, truncated, info


        # 보상 업데이트 (Reward)
        reward, reward_details = self.GetReward()
        self.total_reward += reward

        # 게임의 목표달성 여부
        terminated = False
        truncated = self.step_count >= self.max_steps

        # -----------------------------------------------------------
        # 4. 정보 반환 (Info) - 로그용 데이터
        # -----------------------------------------------------------
        # info = {
        #     # 1. 게임 진행 상황 (Game)
        #     "game/badges": self.current_badge_count,
        #     "game/level_sum": utils.get_level_sum(self.pyboy),
        #     "game/exploration": len(self.seen_coords),
        #     "game/deaths": self.death_count,

        #     # 2. 회복 세분화 (Heals)
        #     "game/heal_battle": self.heal_battle_count,
        #     "game/heal_field": self.heal_field_count,

        #     # 3. 상태 변수 (Stats)
        #     "stats/step_count": self.step_count,
        #     "stats/total_reward": self.total_reward,
        #     "stats/map_id": cur_map_id,

        #     # 4. 보상 상세 (Rewards)
        #     "reward/badge": reward_details["badge"],
        #     "reward/battle": reward_details["battle"],
        #     "reward/explore": reward_details["explore"],
        #     "reward/exp": reward_details["exp"],
        #     "reward/dmg": reward_details["dmg"],
        #     "reward/dead": reward_details["dead"],
        #     "reward/gemini": reward_details["gemini"],
        #     "reward/penalty": reward_details["penalty"]
        # }

        # 수정 [1단계 전용!!]
        info = {
            # Game
            "game/badges": self.current_badge_count,
            "game/exploration": len(self.seen_coords),
            "stats/step_count": self.step_count,
            "stats/total_reward": self.total_reward,
            "stats/map_id": cur_map_id,

            # Rewards (Stage 1)
            "reward/explore": reward_details.get("explore", 0.0),
            "reward/time_penalty": reward_details.get("time_penalty", 0.0),
            "reward/stuck_penalty": reward_details.get("stuck_penalty", 0.0),
            "reward/badge": reward_details.get("badge", 0.0),
        }

        info.update({
            "game/unique_maps": len(self.visited_maps),
            "game/current_map_id": cur_map_id,
            "game/current_map_hex": int(cur_map_id),  # 텐서보드용(문자열은 안 찍힘)
        })

        # 전투 완전히 종료되면 플래그 해제
        if self.handling_battle and self.pyboy.memory[utils.MEM_BATTLE_TYPE] == 0:
            self.handling_battle = False
        
        return obs, reward, terminated, truncated, info

    # def battle_skip_step(self):
    #     """
    #     전투 중이면: 적 HP를 0으로 만들고 A를 눌러서
    #     승리/경험치/전투종료까지 화면을 밀어준다.
    #     """
    #     battle_type = self.pyboy.memory[utils.MEM_BATTLE_TYPE]
    #     if battle_type == 0:
    #         return

    #     battle_tick = 2

    #     self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #     self.pyboy.tick(battle_tick)
    #     self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    #     self.pyboy.tick(battle_tick)

    #     enemy_max_hp = utils.read_be16(self.pyboy, utils.MEM_ENEMY_MAX_HP)
    #     my_max_hp = utils.read_be16(self.pyboy, utils.MEM_BATTLE_HP_MAX)
    #     enemy_hp = utils.read_be16(self.pyboy, utils.MEM_ENEMY_HP)

    #     self.pyboy.tick(battle_tick)
    #     # 1️⃣ 등장 연출 / 텍스트 단계
    #     if enemy_max_hp == 0 or my_max_hp == 0:
    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #         self.pyboy.tick(battle_tick)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    #         self.pyboy.tick(battle_tick)
    #         return

    #     # 2️⃣ 실제 전투 → 적이 살아있을 때만 HP 0
    #     if enemy_hp > 0:
    #         self.pyboy.memory[utils.MEM_ENEMY_HP] = 0
    #         self.pyboy.memory[utils.MEM_ENEMY_HP + 1] = 0

    #         self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #         self.pyboy.tick(battle_tick)
    #         self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    #         self.pyboy.tick(battle_tick)
    #         return

    #     # 3️⃣ 승리 / 경험치 / 종료 텍스트
    #     self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
    #     self.pyboy.tick(battle_tick)
    #     self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
    #     self.pyboy.tick(battle_tick)

    def battle_skip_step(self):
        """
        [TRAIN 전용]
        전투 중이면:
        - A 입력으로 모든 UI / 메시지를 밀고
        - 적 HP를 0으로 만들어 전투 종료를 '유도'
        (※ 전투가 반드시 끝난다는 보장은 없음)
        """
        if self.pyboy.memory[utils.MEM_BATTLE_TYPE] == 0:
            return

        battle_tick = 2

        # 1️⃣ UI / 텍스트 밀기 (PP, 실패 메시지 대비)
        for _ in range(3):
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.tick(battle_tick)
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.tick(battle_tick)

        # 2️⃣ 적 HP 즉살 (엔진이 전투 종료하게 유도)
        enemy_hp = utils.read_be16(self.pyboy, utils.MEM_ENEMY_HP)
        if enemy_hp > 0:
            self.pyboy.memory[utils.MEM_ENEMY_HP] = 0
            self.pyboy.memory[utils.MEM_ENEMY_HP + 1] = 0

            # 즉살 후 텍스트 한 번 더 밀기
            for _ in range(2):
                self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
                self.pyboy.tick(battle_tick)
                self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
                self.pyboy.tick(battle_tick)


    # def GetReward(self):
    #     """
    #     [통합 보상 함수]
    #     기존 기능: 가중치 시스템, 제자리/시간초과 감점, LLM 보상, 배지 보상
    #     추가 기능: EXP, 레벨업, 전투 승리, 적에게 준 데미지, 기절 패널티, 회복 감지
    #     """

    #     # ------------------------------------------------------------------
    #     # 1. 메모리 데이터 읽기 (Hybrid Logic 적용)
    #     # ------------------------------------------------------------------
    #     try:
    #         # 기본 정보 읽기
    #         cur_money = utils.read_bcd(self.pyboy, utils.MEM_MONEY, 3)  # BCD로 읽기 권장
    #         cur_badges = utils.get_badges(self.pyboy)
    #         cur_level_sum = utils.get_level_sum(self.pyboy)
    #         cur_exp = utils.read_uint24(self.pyboy, utils.MEM_P1_EXP)  # Big Endian
    #         cur_battle_type = self.pyboy.memory[utils.MEM_BATTLE_TYPE]

    #         # 맵 정보 (탐험 보상용)
    #         cur_map_group = self.pyboy.memory[utils.MEM_MAP_GROUP]
    #         cur_map_num = self.pyboy.memory[utils.MEM_MAP_NUMBER]
    #         cur_map_id = (cur_map_group << 8) | cur_map_num

    #         # [추가] 좌표 읽기
    #         cur_x = utils.read_uint8(self.pyboy, utils.MEM_X_POS)
    #         cur_y = utils.read_uint8(self.pyboy, utils.MEM_Y_POS)

    #         # [핵심] 전투 vs 필드 체력 주소 스위칭
    #         if cur_battle_type != 0:
    #             # ⚔️ [전투 중] Active Battle Struct (Big Endian)
    #             cur_hp = utils.read_be16(self.pyboy, utils.MEM_BATTLE_HP_NOW)
    #             cur_max_hp = utils.read_be16(self.pyboy, utils.MEM_BATTLE_HP_MAX)
    #             cur_enemy_hp = utils.read_be16(self.pyboy, utils.MEM_ENEMY_HP)
    #         else:
    #             # 🌿 [필드] Party Struct (Little Endian)
    #             cur_hp = utils.read_uint16(self.pyboy, utils.MEM_P1_HP)
    #             cur_max_hp = utils.read_uint16(self.pyboy, utils.MEM_P1_MAX_HP)
    #             cur_enemy_hp = 0

    #     except Exception as e:
    #         print(f"Error reading memory: {e}")
    #         return 0, {}

    #     # 보상 카테고리 초기화
    #     reward_details = {
    #         "badge": 0, "gemini": 0, "penalty": 0, "stuck": 0,  # 기존
    #         "event": 0, "explore": 0, "battle": 0, "level": 0,
    #         "heal": 0, "exp": 0, "dead": 0, "dmg": 0  # 신규
    #     }

    #     total_step_reward = 0.0

    #     #시간 지체 패널티 추가
    #     # 매 스텝마다 조금씩 감점하여 AI가 빨리 움직이도록 강제함

    #     # penalty = -0.00001
    #     # total_step_reward += penalty
    #     # reward_details["penalty"] += penalty

    #     # =================================================================
    #     # [NEW] Stuck (고착 상태) 패널티 적용
    #     # =================================================================
    #     # 1. 아까 만든 함수를 호출해서, 내가 여기 얼마나 오래 있었는지(count) 가져옵니다.
    #     stuck_count = self.get_current_coord_count_reward()

    #     # 2. 오래 있었을수록 패널티를 세게 때립니다.
    #     # 예: 처음 옴(1) -> -0.005점 (거의 없음)
    #     # 예: 100번째 있음(100) -> -0.5점 (매우 아픔! 빨리 탈출해야 함)
    #     stuck_penalty = stuck_count * -0.0001

    #     # 3. 점수 반영
    #     total_step_reward += stuck_penalty
    #     reward_details["stuck"] += stuck_penalty

    #     #배지 개수에 따른 가중치 가져오기
    #     reward_weights = self.config.get("reward_weights", {})
    #     weights = reward_weights.get(cur_badges, reward_weights.get("default", {}))

    #     #패널티 로직
    #     # # (1) 한 맵에 너무 오래 머무름  **동일하게 AI 무기력 문제 발생**
    #     # if self.steps_on_map > 4096:
    #     #     penalty = -0.1
    #     #     total_step_reward += penalty
    #     #     reward_details["penalty"] += penalty
    #     #
    #     #     if self.steps_on_map == 4097:
    #     #         print("한 맵에 너무 오래 있습니다! (-5점)")
    #     #         total_step_reward -= 5.0
    #     #         reward_details["penalty"] -= 5.0

    #     # # (2) 제자리 걸음 (갇힘 감지) **AI 무기력 문제 발생**
    #     # if cur_battle_type == 0:
    #     #     # 100스텝 데이터가 쌓였을 때만 검사
    #     #     if len(self.coord_history) == 100:
    #     #         unique_coords = len(set(self.coord_history))
    #     #
    #     #         # 기준을 10 -> 3으로 대폭 완화
    #     #         # 이유: 2x2 풀숲(좌표 4개)이나 1x3 복도에서 왔다갔다 하는 건 '의도된 행동'일 수 있음.
    #     #         # 하지만 3개 미만(1~2개)이라는 건 진짜 벽에 박고 있거나 제자리 회전만 한다는 뜻.
    #     #         if unique_coords < 3:
    #     #             print("갇힘 감지 (-0.1점)")
    #     #             penalty = -0.1
    #     #             total_step_reward += penalty
    #     #             reward_details["penalty"] += penalty
    #     #
    #     #             # (선택) 갇혔을 때 coord_history를 비워줘서 연속 감점을 막고 새 출발 기회를 줌
    #     #             self.coord_history.clear()

    #     #보상 로직
    #     #뱃지 획득
    #     if cur_badges > self.prev_badges:
    #         r = 100.0
    #         total_step_reward += r
    #         reward_details["badge"] += r
    #         print(f"배지 획득! ({self.prev_badges} -> {cur_badges})")

       
    #     # [Explore] 진짜 탐험 (새로운 좌표 방문 시 보상)
    #     # step 함수에서 이미 self.seen_coords에 현재 좌표를 추가하고 있습니다.
    #     # 따라서 "이전 스텝의 방문 수"보다 "현재 방문 수"가 늘어났다면 새로운 땅을 밟은 것입니다.

    #     # (주의: 이 로직을 쓰려면 GetReward 부르기 직전에 step함수에서 seen_coords 업데이트 하기 전의 길이를 알아야 함)
    #     # 하지만 더 쉬운 방법은, "현재 좌표가 seen_coords에 없었으면 보상"을 주는 것입니다.
    #     # step 함수 구조상 seen_coords.add가 먼저 일어나므로,
    #     # 로직 순서를 살짝 바꾸거나 아래 방식을 추천합니다.

    #     # -------------------------------------------------------
    #     # [수정 제안] GoldEnv.py의 step 함수 로직과 연동된 방식
    #     # -------------------------------------------------------

    #     # # 현재 위치 (맵, X, Y)
    #     # curr_coord = (cur_map_id, cur_x, cur_y)

    #     # # 만약 이 좌표가 내 기억(seen_coords)에 없다면? -> 새로운 땅이다!
    #     # if curr_coord not in self.seen_coords:
    #     #     r = 0.0001 * weights.get("exploration", 1.0)  # 작은 보상을 줌 (티끌 모아 태산)
    #     #     total_step_reward += r
    #     #     reward_details["explore"] += r
    #     #     # (중요) 보상을 줬으니 기록에 추가
    #     #     # 수정 (지우기)
    #     #     # self.seen_coords.add(curr_coord)

    #     # 수정
    #     # [Explore] 신규 좌표 보상 (step에서 판별한 값 사용)
    #     if self.is_new_coord:
    #         r = 0.0001 * weights.get("exploration", 1.0)
    #         total_step_reward += r
    #         reward_details["explore"] += r


    #     # # [Explore] 새로운 맵 진입 **기존 코드**
    #     # if cur_map_id != self.prev_map_id:
    #     #     r = 1.0 * weights.get("explore", 1.0)
    #     #     total_step_reward += r
    #     #     reward_details["explore"] += r
    #     #     # 맵 이동 시 체류 시간 초기화는 Step 함수 등에서 처리한다고 가정

    #     #경험치 보상
    #     if cur_exp > self.prev_exp:
    #         # 경험치는 숫자가 크므로 0.001 곱함
    #         r = (cur_exp - self.prev_exp) * 0.001 * weights.get("exp", 1.0)
    #         total_step_reward += r
    #         reward_details["exp"] += r

    #     #레벨업 보상
    #     if cur_level_sum > self.prev_level_sum:
    #         r = (cur_level_sum - self.prev_level_sum) * 1.0 * weights.get("level", 1.0)
    #         total_step_reward += r
    #         reward_details["level"] += r
    #         print(f"레벨 업! (Total: {cur_level_sum})")

    #     #Dmg 보상
    #     if self.prev_battle_type != 0 and cur_battle_type != 0:
    #         if self.prev_enemy_hp > cur_enemy_hp:
    #             dmg = self.prev_enemy_hp - cur_enemy_hp
    #             r = dmg * 0.001 * weights.get("battle", 1.0)
    #             total_step_reward += r
    #             reward_details["dmg"] += r

    #     #사망 페널티
    #     # if self.prev_hp > 0 and cur_hp == 0:
    #     #     r = -0.1 * weights.get("dead", 1.0)
    #     #     total_step_reward += r
    #     #     reward_details["dead"] += r
    #     # 사망 페널티 (수정)
    #     if self.prev_hp > 0 and cur_hp == 0:
    #         if self.is_train:
    #             pass  # 학습 중에는 무시
    #         else:
    #             r = -0.1 * weights.get("dead", 1.0)
    #             total_step_reward += r
    #             reward_details["dead"] += r


    #     #전투 보상
    #     # 추가
    #     # 전투 보상
    #     if self.is_train and self.prev_battle_type != 0 and cur_battle_type == 0:
    #         total_step_reward += 1.0

    #     # 조건: 전투가 끝났는데(전투->필드) + 적의 체력이 0이었거나 + 경험치가 올랐어야 함
    #     if self.prev_battle_type != 0 and cur_battle_type == 0:
    #         # 1. 진짜 승리 (적 체력이 0이 됨 OR 경험치를 얻음)
    #         # (PyBoy 메모리 타이밍 이슈로 적 체력 0이 감지 안 될 수도 있으니 경험치 증가도 같이 봅니다)
    #         if self.prev_enemy_hp == 0 or cur_exp > self.prev_exp:
    #             r = 1.0 * weights.get("battle", 1.0)
    #             total_step_reward += r
    #             reward_details["battle"] += r
    #             print("전투 승리!")


    #         # # 2. 도망침 (내 체력은 있는데 적 체력도 남아있고 경험치도 그대로임)
    #         # else:
    #         #     total_step_reward -= 0.1
    #         #     print("도망 패널티 (-0.1점)")
    #         #     # 여기서 선택할 수 있습니다.
    #         #     # 옵션 A: 그냥 보상 없음 (0점) -> 추천
    #         #     # 옵션 B: 도망치지 말라고 약간의 감점 (-0.1점)
    #         #     pass

    #     #회복 보상
    #     if cur_hp > self.prev_hp:
    #         #전투 중일 때
    #         if cur_battle_type != 0 and self.prev_hp > 0:
    #             diff = cur_hp - self.prev_hp
    #             r = diff * 0.001 * weights.get("heal", 1.0)
    #             total_step_reward += r
    #             reward_details["heal"] += r
    #             self.heal_battle_count += 1
    #         #필드일 때
    #         elif cur_battle_type == 0:
    #             diff = cur_hp - self.prev_hp
    #             r = diff * 0.01 * weights.get("heal", 1.0)
    #             total_step_reward += r
    #             reward_details["heal"] += r
    #             self.heal_field_count += 1

    #     #기존 필드일 때의 체력 변화
    #     # if cur_battle_type == 0 and cur_hp > self.prev_hp:
    #     #     r = 0.1 * weights.get("heal", 1.0)
    #     #     total_step_reward += r
    #     #     reward_details["heal"] += r

    #     #[Gemini] LLM 코치 보상 (기존 로직 유지)
    #     if self.coach and (self.step_count % self.coach_interval == 0):
    #         obs = self.GetObs()

    #         # --- [추가] LLM에게 떠먹여 줄 정보 포장 ---
    #         game_status = {
    #             "Location ID": cur_map_id,
    #             "Badges": cur_badges,
    #             "Battle Mode": "Yes" if cur_battle_type != 0 else "No",
    #             "My HP": f"{cur_hp}/{cur_max_hp}",
    #             "Enemy HP": f"{cur_enemy_hp}" if cur_battle_type != 0 else "None",
    #             "Level Sum": cur_level_sum,
    #             "Money": cur_money
    #         }
    #         # ----------------------------------------
    #         # LLM에게 현재까지의 상황과 점수를 주고 평가받음
    #         raw_score, reason = self.coach.evaluate_screen(obs, total_step_reward, game_status)

    #         if raw_score != 0:
    #             bonus = raw_score * weights.get("gemini", 1.0)
    #             total_step_reward += bonus
    #             reward_details["gemini"] += bonus
    #             print(f"LLM 평가: \"{reason}\" -> {raw_score}점 (+{bonus:.2f})")

    #     # 5. 내부 변수 업데이트
    #     self.prev_money = cur_money
    #     self.prev_badges = cur_badges
    #     self.prev_exp = cur_exp
    #     self.prev_level_sum = cur_level_sum
    #     self.prev_hp = cur_hp
    #     self.prev_max_hp = cur_max_hp
    #     self.prev_battle_type = cur_battle_type
    #     self.prev_enemy_hp = cur_enemy_hp
    #     self.prev_map_id = cur_map_id

    #     return total_step_reward, reward_details
    def GetReward(self):
        """
        [Curriculum Stage 1]
        목적: '길 찾기'만 학습하는 순수 탐험 보상 함수
        - 전투 / 경험치 / 레벨 / 회복 / LLM: 전부 무시
        - 새로운 좌표 방문 + 생존 압박만 존재
        """

        # -------------------------------
        # 기본 리턴 구조
        # -------------------------------
        reward_details = {
            "explore": 0.0,
            "time_penalty": 0.0,
            "stuck_penalty": 0.0,
            "badge": 0.0,
        }

        total_reward = 0.0

        # -------------------------------
        # 1️⃣ 탐험 보상 (핵심)
        # -------------------------------
        # step()에서 계산된 값 사용
        if self.is_new_coord:
            r = 0.3   # ⭐ 핵심 값 (0.2 ~ 0.5 추천)
            total_reward += r
            reward_details["explore"] += r

            # 새로운 좌표 밟았으니 stuck 카운터 리셋
            self.stuck_steps = 0

        cur_map_grp = utils.read_uint8(self.pyboy, utils.MEM_MAP_GROUP)
        cur_map_num = utils.read_uint8(self.pyboy, utils.MEM_MAP_NUMBER)
        cur_map_id = (cur_map_grp << 8) | cur_map_num

        # 추가
        # __init__에 추가 필요
        # self.visited_maps = set()

        # [GoldEnv.py] GetReward 함수 내부

        # 맵 바뀌었을 때 보상
        if cur_map_id not in self.visited_maps:
            self.visited_maps.add(cur_map_id)

            # 마을/도로는 크게, 실내는 작게
            r = 5.0
            total_reward += r
            reward_details["explore"] += r
            print(f"🗺️ 새 맵 발견! {hex(cur_map_id)} (Group: {cur_map_grp}, Num: {cur_map_num}) -> +{r}")

        IMPORTANT_MAPS = {
            0x1804: 20.0,   # 연두마을 (New Bark Town)
            0x1805: 50.0,   # 29번 도로
            0x1808: 50.0,   # 무궁시티 (Cherrygrove)
            0x1809: 50.0,   # 30번 도로
            0x180A: 100.0,  # 도라지시티 (Violet City)
            0x180B: 150.0,  # 도라지 체육관 (Gym)
        }

        if cur_map_id in IMPORTANT_MAPS and cur_map_id not in self.visited_maps:
            r = IMPORTANT_MAPS[cur_map_id]
            total_reward += r
            reward_details["explore"] += r
            print(f"🚩 목표 맵 도달! {hex(cur_map_id)} -> +{r}")

        # -------------------------------
        # 2️⃣ 시간 패널티 (가만히 있지 마라)
        # -------------------------------
        time_penalty = -0.0005
        total_reward += time_penalty
        reward_details["time_penalty"] += time_penalty

        # -------------------------------
        # 3️⃣ Stuck 패널티 (여기서 죽인다)
        # -------------------------------
        # 이미 step()에서 self.stuck_steps 증가 중
        if self.stuck_steps > 50:
            stuck_penalty = -0.005
            total_reward += stuck_penalty
            reward_details["stuck_penalty"] += stuck_penalty

        if self.stuck_steps > 150:
            stuck_penalty = -0.02
            total_reward += stuck_penalty
            reward_details["stuck_penalty"] += stuck_penalty

        # -------------------------------
        # 4️⃣ 배지 보상 (나침반 역할)
        # -------------------------------
        cur_badges = utils.get_badges(self.pyboy)
        if cur_badges > self.prev_badges:
            r = 30.0
            total_reward += r
            reward_details["badge"] += r
            print(f"🎉 배지 획득! ({self.prev_badges} → {cur_badges})")

        # -------------------------------
        # 5️⃣ 상태 업데이트
        # -------------------------------
        self.prev_badges = cur_badges

        return total_reward, reward_details

    
    # 추가
    def is_in_battle_ui(self):
        """
        전투 메뉴 / 가방 / 텍스트 박스 등
        아직 전투 컨텍스트에 있으면 True
        """
        return self.pyboy.memory[utils.MEM_BATTLE_TYPE] != 0


    # 추가
    def force_battle_win(self):
        # 이미 처리 중이면 재진입 금지
        if self.handling_battle:
            return

        self.handling_battle = True

        # 전투 아닐 때 방어
        if self.pyboy.memory[utils.MEM_BATTLE_TYPE] == 0:
            return

        # 적 HP = 0
        self.pyboy.memory[utils.MEM_ENEMY_HP] = 0
        self.pyboy.memory[utils.MEM_ENEMY_HP + 1] = 0
    
    # def AI_action(self, action_freq):
    #     #버튼 누름
    #     self.pyboy.send_input(self.valid_actions[action_freq])
    #     #녹화에 따라 진행
    #     if self.save_video and not self.headless:
    #         for _ in range(self.action_freq): self.pyboy.tick()
    #     else:
    #         self.pyboy.tick(self.action_freq)
    #     #버튼 뗌
    #     self.pyboy.send_input(self.release_actions[action_freq])
    # 수정
    def AI_action(self, action_idx, ticks=24):
        # 버튼 누름
        self.pyboy.send_input(self.valid_actions[action_idx])
        
        # 지정된 틱만큼 진행 (전투시 1, 평상시 24)
        if self.save_video and not self.headless:
            for _ in range(ticks): 
                self.pyboy.tick(1)
        else:
            self.pyboy.tick(ticks)
            
        # 버튼 뗌
        self.pyboy.send_input(self.release_actions[action_idx])
    
    def GetObs(self):
        #게임 화면 numpy로 받기 / 세로, 가로, 채널(RGB 그리고 투명도)
        screen = self.pyboy.screen.ndarray
        #투명도 제거 -> 게임보이는 투명한 화면이 없음
        if screen.shape[2] == 4: screen = screen[:, :, :3]
        #PyBoy의 순서: 세로, 가로, 채널 -> PPO의 순서: 채널, 세로, 가로
        return np.transpose(screen, (2, 0, 1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #내부 변수 0
        self.step_count = 0
        self.total_reward = 0
        self.seen_coords = set()
        self.steps_on_map = 0
        self.prev_map_id = -1
        self.coord_history.clear()

        self.death_count = 0
        self.heal_battle_count = 0
        self.heal_field_count = 0
        self.prev_coord = None
        self.stuck_steps = 0
        self.is_new_coord = False
        self.visited_maps = set()

        #init부터 다시 시작함
        if self.init_state and os.path.exists(self.init_state):
            with open(self.init_state, "rb") as f: self.pyboy.load_state(f)

        try:
            self.prev_money = utils.read_bcd(self.pyboy, utils.MEM_MONEY, 3)
            self.prev_badges = utils.get_badges(self.pyboy)
            self.current_badge_count = self.prev_badges
            self.prev_level_sum = utils.get_level_sum(self.pyboy)
            self.prev_exp = utils.read_uint24(self.pyboy, utils.MEM_P1_EXP)

            self.prev_battle_type = self.pyboy.memory[utils.MEM_BATTLE_TYPE]

            if self.prev_battle_type != 0:
                self.prev_hp = utils.read_be16(self.pyboy, utils.MEM_BATTLE_HP_NOW)
                self.prev_max_hp = utils.read_be16(self.pyboy, utils.MEM_BATTLE_HP_MAX)
                self.prev_enemy_hp = utils.read_be16(self.pyboy, utils.MEM_ENEMY_HP)
            else:
                self.prev_hp = utils.read_uint16(self.pyboy, utils.MEM_P1_HP)
                self.prev_max_hp = utils.read_uint16(self.pyboy, utils.MEM_P1_MAX_HP)
                self.prev_enemy_hp = 0
            # 맵 ID 초기화
            grp = self.pyboy.memory[utils.MEM_MAP_GROUP]
            num = self.pyboy.memory[utils.MEM_MAP_NUMBER]
            self.prev_map_id = (grp << 8) | num
        except Exception as e:
            print(f"Reset Error (Init variables set to 0): {e}")
            self.prev_money = 0
            self.prev_badges = 0
            self.prev_level_sum = 0
            self.prev_exp = 0
            self.prev_hp = 0
            self.prev_battle_type = 0
            self.prev_enemy_hp = 0

        return self.GetObs(), {}

    def get_current_coord_count_reward(self):
        # 기록이 없으면 0 리턴
        if not self.coord_history:
            return 0

        # 가장 최근 위치 (방금 step에서 추가한 위치)
        current_pos = self.coord_history[-1]

        # 최근 100스텝 중 현재 위치와 동일한 좌표가 몇 개인지 셈
        count = self.coord_history.count(current_pos)

        # 예: 처음 방문이면 1, 계속 벽에 박고 있어서 100번 다 여기였으면 100
        return count

    def close(self):
        #파이썬 프로그램을 끌 때 실행.
        if self.pyboy: self.pyboy.stop()