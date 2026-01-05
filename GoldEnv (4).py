import os
import uuid
import json
from pathlib import Path

import math
import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import mediapy as media 
from einops import repeat

from gymnasium import Env, spaces

# global_map.py가 같은 폴더에 있어야 합니다.
from global_map import local_to_global, GLOBAL_MAP_SHAPE, get_map_id_from_mem

event_flags_start = 0xD7B7
event_flags_end = 0xD8B6

class GoldEnv(Env):
    def __init__(self, config=None):
        self.s_path = Path(config["session_path"])
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = True # Colab용 Headless 강제 설정
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
        self.s_path.mkdir(exist_ok=True, parents=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        
        try:
            with open("map_data.json","r",encoding="utf-8") as f:
                _regions = json.load(f)["regions"]
        except FileNotFoundError:
            print("⚠️ map_data.json을 찾을 수 없습니다.")
            raise

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
        
        self.essential_map_locations = {
            _name_to_id[name]: i for i, name in enumerate(_essential_order) if name in _name_to_id
        }

        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        try:
            with open("events.json") as f:
                event_names = json.load(f)
            self.event_names = event_names
        except FileNotFoundError:
             self.event_names = {}

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=0, high=100, shape=(6,), dtype=np.uint8),
                "badges": spaces.MultiBinary(8),
                "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
                "map": spaces.Box(low=0, high=255, shape=(
                    self.coords_pad * 4, self.coords_pad * 4, 1), dtype=np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks)
            }
        )

        self.pyboy = PyBoy(
            config["gb_path"],
            window="null",
        )

        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        
        self.last_enemy_hp = 0  
        self.has_reset_exploration = False 

    def reset(self, seed=None, options={}):
        self.seed = seed
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

        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(event_flags_start, event_flags_end)
        ])

        self.current_event_flags_set = {}
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        self.last_enemy_hp = self.read_hp(0xD0FF)
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

        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()]),
            "level": np.array(self.get_party_levels(), dtype=np.uint8),
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

        # === 경험치 보상 로직 ===
        current_exp = self.read_party_exp()
        if current_exp > self.last_exp:
            exp_gain = current_exp - self.last_exp
            if exp_gain < 5000:
                self.total_exp_reward += math.log(exp_gain + 1) * 0.5
        self.last_exp = current_exp
        
        # === [수정됨] 데미지 보상 로직 (Log Scale + 1) ===
        is_in_battle = self.read_m(0xD116) != 0 
        current_enemy_hp = self.read_hp(0xD0FF)
        enemy_max_hp = self.read_hp(0xD101)
        
        if is_in_battle and enemy_max_hp > 0:
            if current_enemy_hp < self.last_enemy_hp and self.last_enemy_hp <= enemy_max_hp:
                damage = self.last_enemy_hp - current_enemy_hp
                if 0 < damage <= enemy_max_hp:
                    # [변경] 로그6 적용 + 고정값 1
                    log_dmg = math.log(damage, 6)
                    calc_rew = max(0, log_dmg)
                    self.total_dmg_reward += (calc_rew + 1)
                    
        # === 도감 획득 시 탐험 리셋 로직 ===
        mem_val = self.read_m(0xD88E)
        is_pokedex_event_done = (mem_val >> 5) & 1
        
        if is_pokedex_event_done == 1 and not self.has_reset_exploration:
            if self.print_rewards:
                print(f"\n📢 [Step {self.step_count}] 도감 획득 확인! 탐험 보상을 초기화합니다.")
            self.seen_coords = {} 
            self.has_reset_exploration = True
            
        if is_in_battle:
            self.last_enemy_hp = current_enemy_hp
        else:
            self.last_enemy_hp = 0

        self.party_size = self.read_m(0xDA22)
        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()
        self.step_count += 1

        info = {
            "x": self.agent_stats[-1]["x"],
            "y": self.agent_stats[-1]["y"],
            "map_id": self.agent_stats[-1]["map"],
            "stats_level_sum": float(self.get_levels_sum()),
            "stats_badges": float(self.get_badges()),
            "stats_explore": float(len(self.seen_coords)),
            "stats_deaths": float(self.died_count),
            "reward_total": float(self.total_reward),
            "reward_explore": float(self.progress_reward['explore']),
            "reward_level": float(self.progress_reward['level']),
            "reward_badge": float(self.progress_reward['badge']),
            "reward_event": float(self.progress_reward['event']),
            "reward_heal": float(self.progress_reward['heal']),
            "reward_exp": float(self.progress_reward['exp']),
            "reward_dmg": float(self.progress_reward['dmg'])
        }

        if self.save_video and not step_limit_reached:
             self.add_video_frame()

        if step_limit_reached and self.save_video:
             self.close_video()

        return obs, new_reward, False, step_limit_reached, info

    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        press_step = 8
        self.pyboy.tick(press_step)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1)
        self.pyboy.tick(1)

    def append_agent_stats(self, action):
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        map_id = self.get_verified_map_id()
        if map_id == -1: map_id = -1

        levels = self.get_party_levels()
        
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_id,
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
        
        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        
        full_name = base_dir / f"full_reset_{self.reset_count}_id{self.instance_id}.mp4"
        
        self.full_frame_writer = media.VideoWriter(
            str(full_name), (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()

    def add_video_frame(self):
        if self.full_frame_writer:
             self.full_frame_writer.add_image(
                self.render(reduce_res=False)[:, :, 0]
             )

    def close_video(self):
        if self.full_frame_writer:
            self.full_frame_writer.close()
            self.full_frame_writer = None

    def get_game_coords(self):
        return (self.read_m(0xD20D), self.read_m(0xD20E), self.read_m(0xDA00), self.read_m(0xDA01))

    def get_verified_map_id(self):
        _, _, map_group, map_n = self.get_game_coords()
        current_raw_id = get_map_id_from_mem(map_group, map_n)

        if current_raw_id == self.pending_map_id:
            self.map_stability_count += 1
        else:
            self.pending_map_id = current_raw_id
            self.map_stability_count = 0

        if self.map_stability_count >= 3:
            self.stable_map_id = current_raw_id

        return self.stable_map_id if self.stable_map_id != -1 else current_raw_id

    def update_seen_coords(self):
        if self.read_m(0xD116) == 0:
            x_pos, y_pos, map_group, map_number = self.get_game_coords()
            map_n = get_map_id_from_mem(map_group, map_number)
            
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        map_n = get_map_id_from_mem(map_group, map_number)
        
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        return 0 if count < 600 else 1

    def get_global_coords(self):
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        if map_group < 24:
             return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
        
        map_id = get_map_id_from_mem(map_group, map_number)
        return local_to_global(y_pos, x_pos, map_id)

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
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
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )
        new_step = new_total - self.total_reward
        self.total_reward = new_total
        return new_step

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        return done

    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_party_levels(self):
        return [
            self.read_m(a) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]
        ]

    def get_levels_sum(self):
        # 스타팅 레벨(5) 뺌
        return max(sum(self.get_party_levels()) - 5, 0)

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

    def get_badges(self):
        return self.bit_count(self.read_m(0xD57C))

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xDA23, 0xDA24, 0xDA25, 0xDA26, 0xDA27, 0xDA28]
        ]

    def read_party_exp(self):
        h = self.read_m(0xDA32)
        m = self.read_m(0xDA33)
        l = self.read_m(0xDA34)
        return (h << 16) | (m << 8) | l

    def get_all_events_reward(self):
        return max(
            sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
            ])
            - self.base_event_flags,
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        badge_count = self.get_badges()
        level_sum = self.get_levels_sum()
        allowed_level_cap = 9 + (badge_count * 9)
        
        if level_sum > allowed_level_cap:
            explore_plus = 5
            grinding_penalty = 0
        else:
            explore_plus = 1
            grinding_penalty = 1.0

        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 10,
            "level": self.reward_scale * self.get_levels_reward() * 5.0 * grinding_penalty, 
            "heal": self.reward_scale * self.total_healing_rew * 2,
            "exp": self.reward_scale * self.total_exp_reward * 0.1 * grinding_penalty,
            "dead": self.reward_scale * self.died_count * -0.5,
            "badge": self.reward_scale * self.get_badges() * 20,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.15 * explore_plus,
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05,
            "dmg": self.reward_scale * self.total_dmg_reward * 0.07 * grinding_penalty
        }
        return state_scores

    def update_max_op_level(self):
        opponent_level = self.read_m(0xD0FC)
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

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

    def bit_count(self, bits):
        return bin(bits).count("1")

    def update_map_progress(self):
        verified_map_id = self.get_verified_map_id()
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(verified_map_id))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1
        
    def check_manual_control(self):
        pass