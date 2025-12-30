import uuid
import json
from pathlib import Path

import math
import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
# from pyboy.logger import log_level
import mediapy as media
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

        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()

        # [새로 추가] === 경험치 증가 보상 로직 ===
        current_exp = self.read_party_exp()
        
        # 1. 경험치가 늘어났는가? (전투 승리 등)
        if current_exp > self.last_exp:
            exp_gain = current_exp - self.last_exp
            
            # 2. 비정상적으로 큰 값(예: 고레벨 포켓몬과 교체 버그) 방지용 리미트
            # 초반 야생 포켓몬은 많이 줘봤자 100~500 exp입니다.
            if exp_gain < 5000:
                # 가중치 0.1: 경험치 10 얻으면 보상 1점 (취향껏 조절하세요)
                self.total_exp_reward += math.log(exp_gain + 1) * 0.5
        
        # 3. 현재 경험치를 과거 경험치로 갱신 (줄어든 경우에도 여기서 갱신되므로 페널티 없음)
        self.last_exp = current_exp
        # =======================================

        self.party_size = self.read_m(0xDA22)


        self.party_size = self.read_m(0xDA22)

        new_reward = self.update_reward()

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()
        
        # [수정] info 딕셔너리에 게임 통계 정보를 담습니다.
        # SB3는 이 info를 수집해서 Callback으로 전달해줍니다.
        info = {
            "x": self.agent_stats[-1]["x"],
            "y": self.agent_stats[-1]["y"],
            "map_id": self.agent_stats[-1]["map"],
            "stats_level_sum": self.get_levels_sum(),     # 포켓몬 레벨 총합
            "stats_badges": self.get_badges(),            # 배지 개수
            "stats_explore": len(self.seen_coords),       # 탐험한 좌표 개수
            "stats_deaths": self.died_count,              # 사망 횟수
            "stats_heals": self.total_healing_rew         # 힐링 점수 (꼼수 확인용)
        }

        # self.step_count += 1 (기존 코드)

        # return 값의 마지막 {}를 info로 교체
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
        if self.save_video and self.fast_video:
            self.add_video_frame()

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

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" {key}: {val:5.2f}"
            prog_string += f" sum: {self.total_reward:5.2f}"
            print(f"\r{prog_string}", end="", flush=True)

        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:, :, 0],
            )

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_explore_map.jpeg"
                    ),
                    obs["map"][:, :, 0],
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full_explore_map.jpeg"
                    ),
                    self.explore_map,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False)[:, :, 0],
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

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
        starter_additional_levels = 4
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
        # 0xDA30 ~ 0xDA32: 파티 1번 포켓몬의 현재 EXP (Big Endian)
        h = self.read_m(0xDA30)
        m = self.read_m(0xDA31)
        l = self.read_m(0xDA32)
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
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 5,
            "level": self.reward_scale * self.get_levels_reward() * 4.0, 
            "heal": self.reward_scale * self.total_healing_rew * 2,
            #"op_lvl": self.reward_scale * self.update_max_op_level() * 0.2,
            "exp": self.reward_scale * self.total_exp_reward,
            "dead": self.reward_scale * self.died_count * -0.1,
            "badge": self.reward_scale * self.get_badges() * 10,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1,
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05
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