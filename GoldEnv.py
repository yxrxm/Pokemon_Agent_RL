import os
import uuid
import json
from pathlib import Path

import math
import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
# from pyboy.logger import log_level
# import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

# [추가] 맵 ID 변환 함수 임포트
from global_map import local_to_global, GLOBAL_MAP_SHAPE, get_map_id_from_mem

event_flags_start = 0xD7B7
event_flags_end = 0xD8B6  # expand for SS Anne # old - 0xD7F6


# museum_ticket = (0xD754, 0)

class GoldEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.pending_map_id = -1
        self.stable_map_id = -1
        self.map_stability_count = 0
        self.frame_stacks = 3
        self.explore_weight = (
            1 if "explore_weight" not in config else config["explore_weight"]
        )
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        with open("map_data.json","r",encoding="utf-8") as f:
            _regions = json.load(f)["regions"]
        _name_to_id = {r["name"]: int(r["id"]) for r in _regions}

        _essential_order = [
            "New Bark Town","Cherrygrove City","Route 29","Route 30","Violet City",
            "Sprout Tower","Route 31","Route 32","Azalea Town","Ilex Forest",
            "Goldenrod City","National Park","Ecruteak City","Burned Tower",
            "Tin Tower (Bell Tower)","Route 38","Olivine City","Olivine Lighthouse",
            "Cianwood City","Whirl Islands","Route 42","Mahogany Town","Lake of Rage",
            "Mt. Mortar","Ice Path","Blackthorn City","Dragon's Den",
            "Victory Road (Johto)","Indigo Plateau",
        ]
        _missing = [n for n in _essential_order if n not in _name_to_id]
        if _missing:
            # raise ValueError(f"map_data.json에 없는 지명: {_missing}")
            pass

        self.essential_map_locations = {
            _name_to_id[name]: i for i, name in enumerate(_essential_order) if name in _name_to_id
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            # WindowEvent.PRESS_BUTTON_START, ------------------------------------------ START 꼼수 방지 및 초반 학습을 위한 주석 처리
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            # WindowEvent.RELEASE_BUTTON_START
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.enc_freqs = 8

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad * 4, self.coords_pad * 4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks)
            }
        )

        head = "null" if config["headless"] else "SDL2"

        # log_level("ERROR")
        self.pyboy = PyBoy(
            config["gb_path"],
            # debugging=False,
            # disable_input=False,
            window=head,
        )

        # self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)
        
        self.last_enemy_hp = 0  # 적의 이전 체력 기억
        self.has_reset_exploration = False # 도감 이벤트 때 탐험 리셋 했나요?
        self.exploration_offset = 0.0 # 도감을 받을 때 탐험 점수 저장 변수

        self.exp_snapshot = None
        self.dmg_snapshot = None
        self.level_snapshot = None
        self.heal_snapshot = None

        # [NEW] 야생 노가다로 쌓은 '거품 점수'를 기록할 변수들
        self.exp_deduction = 0.0
        self.dmg_deduction = 0.0
        self.level_deduction = 0.0
        self.heal_deduction = 0.0

        self.is_combat_frozen = False # 전투 관련 리워드 동결됐냐?

    def reset(self, seed=None, options={}):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()
        self.pending_map_id = -1
        self.stable_map_id = -1
        self.map_stability_count = 0

        self.last_exp = self.read_party_exp()
        self.total_exp_reward = 0

        self.agent_stats = []

        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)

        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.total_dmg_reward = 0
        self.total_gain_money = 0

        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        # experiment! 
        # self.max_steps += 128

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.last_enemy_hp = self.read_hp(0xD0FF) # 시작할 때 적 체력 읽기 (보통 0이겠죠)
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2, 2, 1))
            ).astype(np.uint8)
        return game_pixels_render

    def _get_obs(self):

        screen = self.render()

        self.update_recent_screens(screen)

        # normalize to approx 0-1
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
        ])

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD57C):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions
        }

        return observation

    def step(self, action):

        self.step_count += 1

        # if self.step_count%10 == 0:
        #     print(f"{self.read_m(0xD116)}입니다.") #------------------------------------------------------------------- D116 0/비전투 1/야생 2/트레이너

        # 일시정지-------------------------------------
        self.check_manual_control()
        # 일시정지-------------------------------------

        # if self.save_video and self.step_count == 0:
        #     self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        self.update_heal_reward()

        # === 경험치 보상 로직 ===
        current_exp = self.read_party_exp()
        if current_exp > self.last_exp:
            exp_gain = current_exp - self.last_exp
            if exp_gain < 5000:
                self.total_exp_reward += math.log(exp_gain + 1) * 0.5
        self.last_exp = current_exp
        # =======================
        
         # === [개선됨] 데미지 보상 로직 ===
        
        # 1. 전투 여부 확인
        is_in_battle = self.read_m(0xD116) != 0 
        
        current_enemy_hp = self.read_hp(0xD0FF)
        enemy_max_hp = self.read_hp(0xD101)
        
        if is_in_battle and enemy_max_hp > 0:
            
            # 체력이 줄었는지 확인
            if current_enemy_hp < self.last_enemy_hp and self.last_enemy_hp <= enemy_max_hp:
                damage = self.last_enemy_hp - current_enemy_hp
                
                # [핵심 변경] 고정값(100) 대신 '적 최대 체력'을 한계선으로 설정
                # "데미지가 0보다 크고, 적의 최대 체력보다는 작거나 같아야 한다"
                if 0 < damage <= enemy_max_hp:
                    self.total_dmg_reward += damage
                    
        # =================================================================
        # [추가] 도감/미스터리알 획득(0xD88E, bit 5) 시 탐험 초기화 로직
        # =================================================================
        
        # 1. 현재 메모리 값 읽기 (님이 만든 read_m 사용)
        mem_val = self.read_m(0xD88E)
        
        # 2. 5번째 비트가 1인지 확인 (비트 연산)
        # (값 >> 5) & 1 은 5번째 비트만 쏙 빼서 0인지 1인지 보는 겁니다.
        is_pokedex_event_done = (mem_val >> 5) & 1
        
        # 3. 이벤트가 완료됐고(1), 아직 리셋을 안 했다면(False) 실행
        if is_pokedex_event_done == 1 and not self.has_reset_exploration:
            
            if self.print_rewards: # 로그 설정이 켜져 있다면 출력
                print(f"\n📢 [Step {self.step_count}] 도감 획득 확인! 탐험 보상을 초기화합니다. (Backtracking 유도)")
            
            # ---------------------------------------------------------------------
            # [핵심] 현재 적용 중인 '탐험 가중치(explore_plus)'를 여기서도 계산해야 함
            # (get_game_state_reward 함수와 로직이 100% 일치해야 점수 증발이 없음)
            # ---------------------------------------------------------------------
            
            # A. 필요한 변수들 읽기
            badge_count = self.get_badges()
            level_sum = self.get_levels_sum()
            # 레벨캡 공식 (사용자님 설정에 맞게 8 또는 9 확인 필요)
            allowed_cap = 9 + (badge_count * 8) 
            
            # B. 트레이너 배틀 여부 확인 (0xD119: 0이면 야생, >0이면 트레이너)
            trainer_class = self.read_m(0xD119)
            is_trainer_battle = (trainer_class > 0)

            # C. 가중치 결정 (리워드 함수와 똑같이!)
            # "레벨이 너무 높고(Cap 초과) + 야생이다" -> 탐험 가중치 적용 중인 상태
            if level_sum > allowed_cap and not is_trainer_battle:
                current_explore_plus = 5.0  # ⚠️ 중요: 리워드 함수에서 설정한 값(5 or 10)과 똑같이 맞추세요!
            else:
                current_explore_plus = 1.0

            # 현재까지 쌓아온 탐험 점수
            current_explore_val = self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1 * current_explore_plus

            # 2. 적립금(offset)에 추가 (이러면 총점은 유지됨)
            self.exploration_offset += current_explore_val
           
            # [핵심] 방문했던 좌표 기록을 싹 비웁니다.
            # 이제부터는 아는 길도 '초행길' 취급을 받아 점수를 줍니다.
            self.seen_coords = {} 
            
            # (선택 사항) 리셋 축하 보너스 점수 (동기 부여용)
            # self.total_reward += 5.0 
            
            # [중요] 중복 실행 방지 잠금
            self.has_reset_exploration = True
            
        # =================================================================

        # 상태 업데이트
        if is_in_battle:
            self.last_enemy_hp = current_enemy_hp
        else:
            self.last_enemy_hp = 0

        self.party_size = self.read_m(0xDA22)
        
        # 여기서 보상이 갱신됨 (self.progress_reward 딕셔너리에 항목별 점수가 들어있음)
        new_reward = self.update_reward()

        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()

        # [수정됨] info에 '총 보상'과 '세부 보상 항목'을 모두 추가합니다.
        info = {
            "x": self.agent_stats[-1]["x"],
            "y": self.agent_stats[-1]["y"],
            "map_id": self.agent_stats[-1]["map"],
            "stats_level_sum": float(self.get_levels_sum()),
            "stats_badges": float(self.get_badges()),
            "stats_explore": float(len(self.seen_coords)),
            "stats_deaths": float(self.died_count),
            
            # --- 추가된 보상 로그 ---
            "reward_total": float(self.total_reward),           # 현재 총 보상 합계
            "reward_explore": float(self.progress_reward['explore']), # 탐험 보상 점수
            "reward_level": float(self.progress_reward['level']),     # 레벨업 보상 점수
            "reward_badge": float(self.progress_reward['badge']),     # 배지 보상 점수
            "reward_event": float(self.progress_reward['event']),     # 이벤트(스토리) 보상 점수
            "reward_heal": float(self.progress_reward['heal']),       # 회복 보상 점수
            "reward_exp": float(self.progress_reward['exp']),         # 경험치 보상 점수
            "reward_dmg": float(self.progress_reward['dmg'])          # 데미지 보상 점수
        }

        return obs, new_reward, False, step_limit_reached, info

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        # if self.save_video and self.fast_video:
        #     self.add_video_frame()

    def append_agent_stats(self, action):
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        
        # [수정] 로그에도 검증된 ID를 기록 (이제 로그에 18(진청)이 안 찍힙니다)
        map_id = self.get_verified_map_id()

        # 실내나 미확인 맵은 -1로 통일
        if map_id == -1: map_id = -1

        levels = [
            self.read_m(a) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_id,  # 검증된 ID
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xDA22),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "total_reward": self.total_reward
            }
        )

    # def start_video(self):

    #     if self.full_frame_writer is not None:
    #         self.full_frame_writer.close()
    #     if self.model_frame_writer is not None:
    #         self.model_frame_writer.close()
    #     if self.map_frame_writer is not None:
    #         self.map_frame_writer.close()

    #     base_dir = self.s_path / Path("rollouts")
    #     base_dir.mkdir(exist_ok=True)
    #     full_name = Path(
    #         f"full_reset_{self.reset_count}_id{self.instance_id}"
    #     ).with_suffix(".mp4")
    #     model_name = Path(
    #         f"model_reset_{self.reset_count}_id{self.instance_id}"
    #     ).with_suffix(".mp4")
    #     self.full_frame_writer = media.VideoWriter(
    #         base_dir / full_name, (144, 160), fps=60, input_format="gray"
    #     )
    #     self.full_frame_writer.__enter__()
    #     self.model_frame_writer = media.VideoWriter(
    #         base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
    #     )
    #     self.model_frame_writer.__enter__()
    #     map_name = Path(
    #         f"map_reset_{self.reset_count}_id{self.instance_id}"
    #     ).with_suffix(".mp4")
    #     self.map_frame_writer = media.VideoWriter(
    #         base_dir / map_name,
    #         (self.coords_pad * 4, self.coords_pad * 4),
    #         fps=60, input_format="gray"
    #     )
    #     self.map_frame_writer.__enter__()

    # def add_video_frame(self):
    #     self.full_frame_writer.add_image(
    #         self.render(reduce_res=False)[:, :, 0]
    #     )
    #     self.model_frame_writer.add_image(
    #         self.render(reduce_res=True)[:, :, 0]
    #     )
    #     self.map_frame_writer.add_image(
    #         self.get_explore_map()
    #     )

    def get_game_coords(self):
        # x, y, map_group, map_number
        return (self.read_m(0xD20D), self.read_m(0xD20E), self.read_m(0xDA00), self.read_m(0xDA01))

    def get_verified_map_id(self):
        # 1. 현재 메모리 읽기
        _, _, map_group, map_n = self.get_game_coords()
        
        # global_map에서 ID 변환 (이미 import 되어 있음)
        current_raw_id = get_map_id_from_mem(map_group, map_n)

        # 2. 값이 흔들리는지 체크 (Debouncing)
        if current_raw_id == self.pending_map_id:
            self.map_stability_count += 1
        else:
            self.pending_map_id = current_raw_id
            self.map_stability_count = 0

        # 3. 3프레임(약 0.05초) 이상 유지되면 진짜 이동으로 인정
        if self.map_stability_count >= 3:
            self.stable_map_id = current_raw_id

        # 아직 불안정하면 과거의 안정된 값 반환
        return self.stable_map_id if self.stable_map_id != -1 else current_raw_id

    # 여기서 부터 해보자---------------------------------------------------------------------------------
    def update_seen_coords(self):
        # if not in battle
        if self.read_m(0xD116) == 0:
            # [수정] 4개 받아서 변환
            x_pos, y_pos, map_group, map_number = self.get_game_coords()
            map_n = get_map_id_from_mem(map_group, map_number)
            
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1

    def get_current_coord_count_reward(self):
        # 4개 값을 받아서
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        # ID로 변환
        map_n = get_map_id_from_mem(map_group, map_number)
        
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def get_global_coords(self):
        # [수정] 4개 받고 변환 및 실내 예외 처리
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        
        # 실내는 매핑 안 함 (중앙 좌표)
        if map_group < 24:
             return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
        
        map_id = get_map_id_from_mem(map_group, map_number)
        
        if map_id == -1:
             # print(f"Unknown Map: Group {map_group}, Number {map_number}")
             pass

        return local_to_global(y_pos, x_pos, map_id)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            # print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0] - self.coords_pad:c[0] + self.coords_pad,
                c[1] - self.coords_pad:c[1] + self.coords_pad
            ]
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)

    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"] * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog["explore"] * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        # done = self.read_hp_fraction() == 0 # end game on loss
        return done

    # def save_and_print_info(self, done, obs):
    #     if self.print_rewards:
    #         prog_string = f"step: {self.step_count:6d}"
    #         for key, val in self.progress_reward.items():
    #             prog_string += f" {key}: {val:5.2f}"
    #         prog_string += f" sum: {self.total_reward:5.2f}"
    #         print(f"\r{prog_string}", end="", flush=True)

    #     if self.step_count % 50 == 0:
    #         plt.imsave(
    #             self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
    #             self.render(reduce_res=False)[:, :, 0],
    #         )

    #     if self.print_rewards and done:
    #         print("", flush=True)
    #         if self.save_final_state:
    #             fs_path = self.s_path / Path("final_states")
    #             fs_path.mkdir(exist_ok=True)
    #             plt.imsave(
    #                 fs_path
    #                 / Path(
    #                     f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
    #                 ),
    #                 obs["map"][:, :, 0],
    #             )
    #             plt.imsave(
    #                 fs_path
    #                 / Path(
    #                     f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
    #                 ),
    #                 self.explore_map,
    #             )
    #             plt.imsave(
    #                 fs_path
    #                 / Path(
    #                     f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
    #                 ),
    #                 self.render(reduce_res=False)[:, :, 0],
    #             )

    #     if self.save_video and done:
    #         self.full_frame_writer.close()
    #         self.model_frame_writer.close()
    #         self.map_frame_writer.close()

    def read_m(self, addr):
        # return self.pyboy.get_memory_value(addr)
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 3 # init이  스타팅을 받고 시작하나?? 4 -> 3 / 4로 했을 땐 스타팅이 6이어도 levelsum이 0
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    #문제점: _get_obs에서는 배지 주소를 0xD57C로 읽는데, get_badges 함수에서는 0xD57D로 읽습니다. 주소가 다릅니다.
    #해결책: wram.asm에서 확인한 정확한 배지 주소로 통일해야 합니다. (DataCrystal 정보로는 성도 배지가 $DCD0, 관동 배지가 $DCD1이었습니다. D57C/D57D가 맞는지 재확인이 필요합니다.)
    def get_badges(self):
        return self.bit_count(self.read_m(0xD57C))  # 성도 지방 뱃지에 한하여,

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xDA23, 0xDA24, 0xDA25, 0xDA26, 0xDA27, 0xDA28]
        ]

    # [새로 추가] 파티 첫 번째 포켓몬의 경험치(3바이트) 읽기
    def read_party_exp(self):
        # 0xDA32 ~ 0xDA34: 파티 1번 포켓몬의 현재 EXP (Big Endian)
        h = self.read_m(0xDA32)
        m = self.read_m(0xDA33)
        l = self.read_m(0xDA34)
        return (h << 16) | (m << 8) | l

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags,
            # - int(self.read_bit([0], museum_ticket[1])),
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        # 1. 현재 스펙 확인

        current_exp_score = self.reward_scale * self.total_exp_reward * 0.1
        current_dmg_score = self.reward_scale * self.total_dmg_reward * 0.05 # (원래 0.01 설정이면 유지)
        current_level_score = self.reward_scale * self.get_levels_reward() * 5.0
        current_heal_score = self.reward_scale * self.total_healing_rew * 2

        badge_count = self.get_badges()
        level_sum = self.get_levels_sum()
        
        # 2. 성장 한계선 설정 (배지 0개면 15레벨, 23/31/39/47/55/63/71
        allowed_level_cap = 9 + (badge_count * 8)
        
        # =================================================================
        # 🕵️‍♂️ [핵심 추가] 지금 누구랑 싸우고 있니?
        # =================================================================
        # 0xD116: 전투 타입 (0 비전투 / 1 야생 / 2 트레이너)
        battle_type = self.read_m(0xD116)
        is_trainer_battle = (battle_type == 2)

        # =================================================================
        # ⚖️ 레벨캡 패널티 적용 로직 (예외 처리 추가)
        # =================================================================
        # [조건 설명]
        # 1. 레벨 총합이 한계선을 넘었고 (level_sum > allowed_level_cap)
        # 2. AND, 지금 싸우는 게 트레이너가 아니라면 (not is_trainer_battle)
        # -> 그때만 보상을 동결(0점)시킨다.

        if level_sum > allowed_level_cap and not is_trainer_battle:
            if not self.is_combat_frozen:
                # 📸 처음 넘는 순간! 각각의 점수를 스냅샷으로 저장
                self.exp_snapshot = current_exp_score - self.exp_deduction
                self.dmg_snapshot = current_dmg_score - self.dmg_deduction
                self.level_snapshot = current_level_score - self.level_deduction
                self.heal_snapshot = current_heal_score - self.heal_deduction
                self.is_combat_frozen = True
            
            # [핵심] 실제 점수가 오르는 족족 차감액(deduction)을 늘려버립니다.
            # 목표: (curr_exp - new_deduction) 값이 항상 frozen_base와 같게 만듦.
            self.exp_deduction = current_exp_score - self.exp_snapshot
            self.dmg_deduction = current_dmg_score - self.dmg_snapshot 
            self.level_deduction = current_level_score - self.level_snapshot
            self.heal_deduction = current_heal_score - self.heal_snapshot

            # # 🧊 보상으로는 '더 이상 오르지 않는 스냅샷 값'을 사용
            # final_exp = self.exp_snapshot
            # final_dmg = self.dmg_snapshot
            # final_level = self.level_snapshot

            explore_plus = 5

        # [상황 B] 레벨캡 미만 or 배지 획득 (해제)
        else:
            self.is_combat_frozen = False
            # self.exp_snapshot = None
            # self.dmg_snapshot = None
            # self.level_snapshot = None

            # [핵심] 여기서는 deduction을 업데이트하지 않습니다! (Freeze Deduction)
            # 즉, 야생에서 쌓은 차감액은 그대로 유지하고, 
            # 트레이너한테서 얻은 추가 점수만 반영됩니다.
            
            # # 🔥 동결 해제! 실제 점수가 그대로 반영됨 (밀린 보상 일시불 지급)
            # final_exp = current_exp_score
            # final_dmg = current_dmg_score
            # final_level = current_level_score
            
            explore_plus = 1
        
        
        # 4. 최종 점수 적용 (실제 점수 - 차감액)
        final_exp = current_exp_score - self.exp_deduction
        final_dmg = current_dmg_score - self.dmg_deduction
        final_level = current_level_score - self.level_deduction
        final_heal = current_heal_score - self.heal_deduction

        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 10,
            "level": final_level, 
            "heal": final_heal,
            #"op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            "exp": final_exp,
            "dead": self.reward_scale * self.died_count * -1,
            "badge": self.reward_scale * self.get_badges() * 20,
            "explore": self.exploration_offset + self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1 * explore_plus,
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05,
            "dmg": final_dmg
        }

        return state_scores

    def update_max_op_level(self):
        # [수정] 골드 버전 주소로 변경
        opponent_level = self.read_m(0xD0FC)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xDA22) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xDA4C, 0xDA7C, 0xDAAC, 0xDADC, 0xDB0C, 0xDB3C]
            # Hp 저장되어있는 주소의 시작점(두 바이트에 거쳐 저장되어있는데, 제미나이가 시작점의 주소만 적으면 된대)
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xDA4E, 0xDA7E, 0xDAAE, 0xDADE, 0xDB0E, 0xDB3E]
            # Max Hp 저장되어있는 주소의 시작점(두 바이트에 거쳐 저장되어있는데, 제미나이가 시작점의 주소만 적으면 된대)
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        # [수정] 날것의 ID 대신 '검증된 ID'를 사용하여 보상 오류 방지
        verified_map_id = self.get_verified_map_id()
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(verified_map_id))


    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1
        
    def read_hp(self, addr):
        h = self.read_m(addr)
        l = self.read_m(addr + 1)
        return (h << 8) | l
    
    # [새로 추가] 외부 파일 체크 및 수동 조작 처리 함수
    def check_manual_control(self):
        # 1. 절대 경로 확인
        # GoldEnv.py가 있는 폴더 위치를 기준으로 잡습니다.
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "agent_enabled.txt")
        
        # [디버깅] 처음 1번만 경로를 출력해서 내가 파일을 어디에 만들어야 하는지 알려줌
        if self.step_count == 0:
            print(f"📂 [DEBUG] 파일 찾는 위치: {file_path}", flush=True)

        try:
            if not os.path.exists(file_path):
                # 파일이 없으면 만들라고 알려줌
                if self.step_count % 100 == 0: # 도배 방지
                    print(f"⚠️ [DEBUG] 파일이 없습니다! 여기 만들어주세요: {file_path}", flush=True)
                return

            # 2. 파일 읽기
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read().strip().lower()

            # [디버깅] 읽은 내용이 'no'일 때만 반응
            if content == "no":
                print(f"🛑 [Manual Mode] 감지됨! (내용: {content}) -> 수동 조작 시작", flush=True)
                
                # 'yes'로 바뀔 때까지 여기서 무한 루프
                while True:
                    self.pyboy.tick() # 게임 화면은 계속 돌아가게 함
                    
                    # 파일 다시 읽기 (혹시 내용을 바꿨나?)
                    try:
                        with open(file_path, "r", encoding='utf-8') as f:
                            new_content = f.read().strip().lower()
                        
                        if new_content != "no":
                            print(f"▶️ [Auto Mode] 재개! (내용이 '{new_content}'로 변경됨)", flush=True)
                            break # 루프 탈출!
                            
                    except:
                        pass
                        
        except Exception as e:
            print(f"❌ [Error] 파일 읽기 오류: {e}", flush=True)