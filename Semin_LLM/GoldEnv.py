#GoldEnv 보상 체계 파일
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
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
        self.prev_money = 0 #돈의 양의 변화를 감지하는 용
        self.prev_badges = 0 #뱃지의 증가(변화)를 감지하는 용

        #통계용 변수
        self.seen_coords = set() #게임 내 탐험 좌표
        self.max_level_sum = 0 #게임에서 만난 최대 레벨
        self.death_count = 0
        self.heal_count = 0
        
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
            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PRESS_BUTTON_SELECT
        ]

        #에이전트가 뗄 때 사용하는 버튼
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_START,
            WindowEvent.RELEASE_BUTTON_SELECT
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
                    obs = self.get_observation()
                    advice = self.coach.ask_advice(obs, "사용자가 직접 조언을 요청했어.")
                    print(f" {advice}")
                else:
                    print("코치 없음.")

            # elif content == "저장":
            #     # 여기에 강제 저장 로직을 넣을 수도 있음 (지금은 로그만)
            #     print("💾 (구현 예정) 현재 상태 강제 저장 요청됨!")
            #
            # elif content.startswith("배율"):
            #     print(f"🔧 가중치 조절 명령: {content}")

            #명령어 수행 후, 파일을 비우기
            with open(cmd_file, "w", encoding="utf-8") as f:
                f.write("")

        #파일이 lock인 경우.
        except Exception as e:
            print(f"에러 발생 : {e}")
            pass
        
#AI의 Step과 Step당의 Update 사항
    def step(self, action):
        self.step_count += 1

        #AI가 움직임.
        self.AI_action(action)

        #현재 뱃지 수 확인
        self.current_badge_count = utils.get_badges(self.pyboy)

        #통계 변수 업데이트
        cur_map = utils.read_uint8(self.pyboy, utils.MEM_MAP_NUMBER)
        cur_x = utils.read_uint8(self.pyboy, utils.MEM_X_POS)
        cur_y = utils.read_uint8(self.pyboy, utils.MEM_Y_POS)
        self.seen_coords.add((cur_map, cur_x, cur_y))

        cur_level_sum = utils.get_level_sum(self.pyboy)
        if cur_level_sum > self.max_level_sum:
            self.max_level_sum = cur_level_sum

        #현재 게임화면을 담아옴.
        obs = self.GetObs()
        #보상 업데이트
        reward, reward_details = self.GetReward()
        self.total_reward += reward

        #게임의 목표달성 여부
        terminated = False #목표 종료 조건을 추가 할 수 있음.
        truncated = self.step_count >= self.max_steps #타임 종료 조건

        #2048 step당 보여줄 정보(Callback 함수) 용도
        info = {
            "badges": self.current_badge_count,
            "step_count": self.step_count, #badge 개수 변화에 따라 스텝에 따른 저장 여부 판단용도
            "exploration": len(self.seen_coords),
            "level_sum": cur_level_sum,
            "deaths": self.death_count,
            "heal": self.heal_count,
            "reward_details": reward_details
        }

        #Gymnasium의 기본 규칙 obs, reward, terminated, truncated, info로 그냥 되어있음.
        return obs, reward, terminated, truncated, info

    def GetReward(self):
        """
        보상 총합과 상세 내역(dict)을 함께 리턴
        """
        reward_details = {
            "badge": 0,
            "gemini": 0,
        }

        try:
            current_money = utils.read_uint16(self.pyboy, utils.MEM_MONEY)
            current_badges = self.current_badge_count
        except:
            return 0, reward_details

        #뱃지 개수에 따른 가중치 받아오기
        reward_weights = self.config.get("reward_weights", {})
        weights = reward_weights.get(current_badges, reward_weights.get("default", {}))

        total_step_reward = 0.0

        #뱃지가 늘었다면 보상을 주기 (조건 보상) 더 추가해도 됨.
        if current_badges > self.prev_badges:
            r = 100.0 #보상 값
            total_step_reward += r
            reward_details["badge"] += r #뱃지로 인한 보상을 구분하기 위함
            print(f"배지 획득! ({self.prev_badges} -> {current_badges})")

        #LLM이 주는 보상
        if self.coach and (self.step_count % self.coach_interval == 0):
            obs = self.GetObs()

            #LLM의 보상을 준 이유를 설명하도록 함
            raw_score, reason = self.coach.evaluate_screen(obs, total_step_reward)

            if raw_score > 0:
                bonus = raw_score * weights.get("gemini", 1.0)
                total_step_reward += bonus
                reward_details["gemini"] += bonus

                # ★ 콘솔에 이유 출력
                print(f"LLM의 이유: \"{reason}\" -> {raw_score}점 (보너스 +{bonus:.2f})")

        #내부 변수 업데이트
        self.prev_money = current_money
        self.prev_badges = current_badges

        return total_step_reward, reward_details

    def AI_action(self, action_freq):
        #버튼 누름
        self.pyboy.send_input(self.valid_actions[action_freq])
        #녹화에 따라 진행
        if self.save_video and not self.headless:
            for _ in range(self.action_freq): self.pyboy.tick()
        else:
            self.pyboy.tick(self.action_freq)
        #버튼 뗌
        self.pyboy.send_input(self.release_actions[action_freq])

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

        #init부터 다시 시작함
        if self.init_state and os.path.exists(self.init_state):
            with open(self.init_state, "rb") as f: self.pyboy.load_state(f)

        try:
            self.prev_money = utils.read_uint16(self.pyboy, utils.MEM_MONEY)
            self.prev_badges = utils.get_badges(self.pyboy)
            self.current_badge_count = self.prev_badges
        except:
            self.prev_money = 0
            self.prev_badges = 0

        return self.GetObs(), {}

    def close(self):
        #파이썬 프로그램을 끌 때 실행.
        if self.pyboy: self.pyboy.stop()