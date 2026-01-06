import uuid  # 535줄 아직 안 고침
import json
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
# from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from global_map import local_to_global, GLOBAL_MAP_SHAPE

event_flags_start = 0xD7B7
event_flags_end = 0xD8B6  # expand for SS Anne # old - 0xD7F6
CAPTURE_REWARD = 500.0  # 상단에 상수로 두는 게 깔끔

# ==== (고속 학습용 reward 상수) ====
# battle 관련 주소는 100% 확정이 아니라서 상수화
ADDR_BATTLE_FLAG = 0xD116   # 네 코드에서 이미 사용하던 전투 플래그 기준 유지
ADDR_ENEMY_LEVEL = 0xD0FC   # 추측: Gen2에서 현재 상대 포켓몬 레벨

# ==== (스토리 클리어 중심 reward 상수) ====
REW_STORY_PER_STAGE   = 3000.0   # 필수 맵 단계 1개 진행당
REW_EVENT_PER_FLAG    = 300.0    # 이벤트 플래그 1개 증가당
REW_BADGE_PER_BADGE   = 5000.0   # 뱃지 1개 획득당 (스토리 핵심이라 매우 크게)
REW_LEVEL_PER_SUM     = 3.0      # 파티 레벨합 1 증가당 (파밍 방지 위해 낮게)
REW_BATTLE_WIN_BASE   = 40.0     # 전투 승리 기본
REW_BATTLE_WIN_LVL    = 0.5      # 상대 레벨 보너스
PEN_BATTLE_STEP       = 0.02     # 전투 1스텝당 시간 패널티
PEN_BATTLE_LOSS       = 120.0    # 전투 패배 패널티

REW_EXPLORE_PER_TILE  = 0.05     # "유니크 좌표 수" 기반 탐험 누적 보상
PEN_STUCK_STEP        = 6.0      # 같은 칸 오래 있을 때 패널티 (강하게)
PEN_DEATH_PER_COUNT   = 500.0    # 죽을 때마다 누적 패널티

REW_HEAL_SCALE        = 4.0      # 힐링 파밍 방지 위해 기존 8 -> 4로

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
        self.stagnation_steps = 0
        self.last_map_progress = 0

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
            raise ValueError(f"map_data.json에 없는 지명: {_missing}")

        self.essential_map_locations = {
            _name_to_id[name]: i for i, name in enumerate(_essential_order)
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
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # load event names
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

        self.pyboy = PyBoy(
            config["gb_path"],
            window=head,
        )

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options={}):
        self.seed = seed
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.init_map_mem()
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
        self.last_died_count = 0
        # GoldEnv.__init__ 안
        self.last_party_count = self.read_m(0xDA22)
        self.total_capture_rew = 0.0



        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}

        self.max_map_progress = 0

        # ====== reward tracking vars (고속학습용) ======
        self.last_event_count = self.get_all_events_reward()
        self.last_badge_count = self.get_badges()
        self.last_level_sum = self.get_levels_sum_safe()
        self.last_map_progress = 0
        self.last_in_battle = False

        self.total_story_rew = 0.0
        self.total_event_rew = 0.0
        self.total_badge_rew = 0.0
        self.total_level_rew = 0.0
        self.total_battle_rew = 0.0
        self.total_explore_rew = 0.0
        self.total_stuck_pen = 0.0
        self.total_death_pen = 0.0

        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2, 2, 1))
            ).astype(np.uint8)
        return game_pixels_render

    def _get_obs(self):
        screen = self.render()
        self.update_recent_screens(screen)

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
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        self.update_heal_reward()
        self.party_size = self.read_m(0xDA22)

        # ===== reward components update (핵심) =====
        self.update_story_reward()
        self.update_event_reward()
        self.update_badge_reward()
        self.update_level_reward()
        self.update_battle_reward()
        self.update_explore_reward()
        self.update_stuck_penalty()
        self.update_death_penalty()
        self.update_capture_reward()


        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()

        step_limit_reached = self.check_if_done()
        obs = self._get_obs()

        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names.keys():
                            self.current_event_flags_set[key] = self.event_names[key]
                        else:
                            print(f"could not find key: {key}")

        self.step_count += 1
        return obs, new_reward, False, step_limit_reached, {}

    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
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
                "event": self.progress_reward.get("event", 0),
                "healr": self.total_healing_rew,
            }
        )

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(
            f"map_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60, input_format="gray"
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:, :, 0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:, :, 0]
        )
        self.map_frame_writer.add_image(
            self.get_explore_map()
        )

    def get_game_coords(self):
        return (self.read_m(0xD20D), self.read_m(0xD20E), self.read_m(0xDA01))

    # ------------------- 탐험/좌표 -------------------
    def update_seen_coords(self):
        if self.read_m(ADDR_BATTLE_FLAG) == 0:
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
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

    # ------------------- reward pretty diff -------------------
    def update_reward(self):
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward
        self.total_reward = new_total
        return new_step

    def group_rewards(self):
        prog = self.progress_reward
        return (
            prog.get("level", 0) * 100 / self.reward_scale,
            self.read_hp_fraction() * 2000,
            prog.get("explore", 0) * 150 / (self.explore_weight * self.reward_scale),
        )

    def check_if_done(self):
        return self.step_count >= self.max_steps - 1

    def update_capture_reward(self):
        """포켓몬 포획 성공 시 보상"""
        try:
            party_count = self.read_m(0xDA22)
        except Exception:
            return

        if party_count > self.last_party_count:
            # 🎉 포획 성공!
            self.total_capture_rew += CAPTURE_REWARD
            self.last_party_count = party_count
            if self.debug:
                print(f"🎯 [CAPTURE] 포켓몬 포획! 파티 수 = {party_count}")

    
    # ------------------- memory reads -------------------
    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end)
            for bit in f"{self.read_m(i):08b}"
        ]

    # ------------------- levels -------------------
    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_sum_safe(self):
        """파티 수만큼만 레벨 읽기 (빈 슬롯 쓰레기값 방지)"""
        party_n = int(self.read_m(0xDA22))
        level_addrs = [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
        party_n = max(0, min(party_n, 6))
        levels = [self.read_m(level_addrs[i]) for i in range(party_n)]
        return sum(levels)

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

    # ------------------- badges / party / events -------------------
    def get_badges(self):
        return self.bit_count(self.read_m(0xD57C))  # 성도 뱃지 카운트(추정)

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xDA23, 0xDA24, 0xDA25, 0xDA26, 0xDA27, 0xDA28]
        ]

    def get_all_events_reward(self):
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ]) - self.base_event_flags,
            0,
        )

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health and self.read_m(0xDA22) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                self.died_count += 1

    # ------------------- HP -------------------
    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xDA4C, 0xDA7C, 0xDAAC, 0xDADC, 0xDB0C, 0xDB3C]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xDA4E, 0xDA7E, 0xDAAE, 0xDADE, 0xDB0E, 0xDB3E]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # ------------------- utils -------------------
    def bit_count(self, bits):
        return bin(bits).count("1")

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        _, _, map_idx = self.get_game_coords()
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    # ===================== Reward helpers (고속학습 핵심) =====================

    def is_in_battle(self):
        return self.read_m(ADDR_BATTLE_FLAG) != 0

    def read_enemy_level(self):
        # 주소가 틀리면 값이 이상할 수 있음. 그땐 ADDR_ENEMY_LEVEL 바꾸기.
        return self.read_m(ADDR_ENEMY_LEVEL)

    # def update_story_reward(self):
    #     """필수 맵 순서 진행 보상 (가장 중요)"""
    #     prog = self.max_map_progress
    #     # if prog > self.last_map_progress:
    #     #     delta = prog - self.last_map_progress
    #     #     self.total_story_rew += REW_STORY_PER_STAGE * delta
    #     #     self.last_map_progress = prog
    #     if prog > self.last_map_progress:
    #         delta = prog - self.last_map_progress
    #         self.total_story_rew += REW_STORY_PER_STAGE * delta
    #         self.last_map_progress = prog
    #     else:
    #         # 🔥 스토리 진전 없는 상태에서 오래 머물면 패널티
    #         self.total_story_rew -= 0.05

    # GoldEnv.py 내부
    def update_story_reward(self):
        prog = self.max_map_progress
        
        if prog > self.last_map_progress:
            self.total_story_rew += REW_STORY_PER_STAGE
            self.last_map_progress = prog
            self.stagnation_steps = 0  # 진전 있으면 초기화
        else:
            self.stagnation_steps += 1
            # 300스텝(약 수십 초) 이상 정체 시 패널티 부과 시작
            if self.stagnation_steps > 300:
                self.total_story_rew -= 0.01 * (self.stagnation_steps // 100)

    def update_event_reward(self):
        """스토리/퀘스트 이벤트 플래그 보상"""
        cur_events = self.get_all_events_reward()
        if cur_events > self.last_event_count:
            delta = cur_events - self.last_event_count
            self.total_event_rew += REW_EVENT_PER_FLAG * delta
            self.last_event_count = cur_events


    def update_badge_reward(self):
        """뱃지는 스토리 핵심 마일스톤 → 매우 큰 보상"""
        cur_badges = self.get_badges()
        if cur_badges > self.last_badge_count:
            delta = cur_badges - self.last_badge_count
            self.total_badge_rew += REW_BADGE_PER_BADGE * delta
            self.last_badge_count = cur_badges


    def update_level_reward(self):
        """레벨업 보상은 낮게 (무한 파밍 방지)"""
        cur_level_sum = self.get_levels_sum_safe()
        if cur_level_sum > self.last_level_sum:
            delta = cur_level_sum - self.last_level_sum
            self.total_level_rew += REW_LEVEL_PER_SUM * delta
            self.last_level_sum = cur_level_sum


    def update_battle_reward(self):
        """
        전투는 '필요한 만큼만' 하도록:
        - 전투 중엔 시간 패널티
        - 승리 보상은 작게
        - 패배 패널티는 확실히
        """
        in_battle = self.is_in_battle()

        if in_battle:
            self.total_battle_rew -= PEN_BATTLE_STEP

        if (self.last_in_battle is True) and (in_battle is False):
            my_hp = self.read_hp_fraction()

            try:
                enemy_lvl = self.read_enemy_level()
            except Exception:
                enemy_lvl = 0

            if my_hp > 0:
                self.total_battle_rew += (REW_BATTLE_WIN_BASE + REW_BATTLE_WIN_LVL * enemy_lvl)
            else:
                self.total_battle_rew -= PEN_BATTLE_LOSS

        self.last_in_battle = in_battle


    def update_explore_reward(self):
        """
        탐험 보상(유니크 좌표 기반).
        이동 자체가 스토리 진행의 기반이므로 전투보다 확실히 이득이 되게 설정.
        """
        self.total_explore_rew = REW_EXPLORE_PER_TILE * len(self.seen_coords)


    def update_stuck_penalty(self):
        """
        같은 칸에 오래 머무르는 행동 강하게 억제.
        get_current_coord_count_reward() == 1 이면
        '현재 좌표가 600스텝 이상 반복' 상태.
        """
        if self.get_current_coord_count_reward() == 1:
            self.total_stuck_pen -= PEN_STUCK_STEP


    # def update_death_penalty(self):
    #     if self.died_count > self.last_died_count:
    #         # 이번 스텝에서 막 죽었을 때만 강한 충격(-500)
    #         current_pen = -500.0
    #         self.last_died_count = self.died_count
    #     else:
    #         # 평소에는 죽음에 대한 공포를 0으로 유지 (다시 나가게 함)
    #         current_pen = 0.0
        
    #     self.total_death_pen = current_pen

    def update_death_penalty(self):
        if not hasattr(self, "last_died_count"):
            self.last_died_count = self.died_count

        if self.died_count > self.last_died_count:
            self.total_death_pen += -50.0
            self.last_died_count = self.died_count


    # ===================== get_game_state_reward 교체 =====================

    def get_game_state_reward(self, print_stats=False):
        """
        누적형 reward dict.
        update_reward()에서 차분을 내므로
        여기 값들은 '총합'이 되어야 함.
        """
        state_scores = {
            # 스토리/진도: 최우선
            "story":  self.reward_scale * self.total_story_rew,

            # 스토리 이벤트/뱃지: 진도 보조축
            "event":  self.reward_scale * self.total_event_rew,
            "badge":  self.reward_scale * self.total_badge_rew,

            # 성장/전투: 수단이므로 낮게
            "level":  self.reward_scale * self.total_level_rew,
            "battle": self.reward_scale * self.total_battle_rew,

            # 힐링 파밍 방지용으로 약간만
            "heal":   self.reward_scale * self.total_healing_rew * REW_HEAL_SCALE,

            # 이동 유도
            "explore": self.reward_scale * self.explore_weight * self.total_explore_rew,

            # 나쁜 행동 억제
            "stuck":  self.reward_scale * self.total_stuck_pen,
            "death":  self.reward_scale * self.total_death_pen,

            "capture": self.total_capture_rew,
        }
        return state_scores
