import os
import uuid
import json
from pathlib import Path

import math
import numpy as np
import matplotlib.pyplot as plt
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from einops import repeat
import cv2

from gymnasium import Env, spaces

# [추가] 맵 ID 변환 함수 임포트
from global_map import local_to_global, GLOBAL_MAP_SHAPE, get_map_id_from_mem

event_flags_start = 0xD7B7
event_flags_end = 0xD8B6 

class GoldEnv(Env):
    def __init__(self, config=None):
        self.s_path = Path(config["session_path"])
        
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
        
        self.s_path.mkdir(exist_ok=True, parents=True)
        
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        
        def load_json_safe(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    return json.load(f)
            except UnicodeDecodeError:
                try:
                    with open(filename, "r", encoding="cp949") as f:
                        return json.load(f)
                except Exception:
                    print(f"⚠️ [경고] {filename} 인코딩 깨짐. 강제로 읽습니다.")
                    with open(filename, "r", encoding="utf-8", errors='ignore') as f:
                        return json.load(f)
            except FileNotFoundError:
                return {} 

        self.essential_map_locations = {} 

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

        self.event_names = load_json_safe("events.json")

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12

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
        
        self.last_enemy_hp = 0  
        self.has_reset_exploration = False 
        self.exploration_offset = 0.0 

        self.accumulated_level_reward = 0.0
        self.last_level_sum = 0.0
        
        self.accumulated_stuck_penalty = 0.0

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
        self.prev_in_battle = False
        self.battle_last_enemy_hp = 0
        self.total_battle_bonus = 0.0
        
        self.visited_maps = set()
        self.total_new_map_reward = 0.0

        self.last_level_sum = self.get_levels_sum()
        self.accumulated_level_reward = self.compute_scaled_level(self.last_level_sum)
        
        self.accumulated_stuck_penalty = 0.0

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
        
        init_map = self.get_verified_map_id()
        if init_map is not None and init_map != -1:
            self.visited_maps.add(init_map)
            
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}          
        self._stuck_last_key = None    
        self._stuck_move_count = 0     

    def render(self, reduce_res=True):
            if reduce_res:
                screen = self.pyboy.screen.ndarray[:, :, 0] 

                resized = cv2.resize(screen, (80, 72), interpolation=cv2.INTER_AREA)
                game_pixels_render = resized[:, :, None].astype(np.uint8)
                
            else:
                game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1].astype(np.uint8)
                
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

    def get_lead_level(self) -> int:
        return int(self.read_m(0xDA49))

    def get_cap_multiplier(self, is_trainer_battle):
        if is_trainer_battle:
            return 1.0

        badge_count = self.get_badges()
        allowed_cap = 11 + (badge_count * 8)

        lead_level = self.get_lead_level()   
        diff = allowed_cap - lead_level     

        if diff >= 3: return 1.2
        if diff == 2: return 0.6
        if diff == 1: return 0.3
        return 0.01

    def compute_scaled_level(self, level_sum):
        explore_thresh = 22
        scale_factor = 4
        if level_sum < explore_thresh:
            return level_sum
        else:
            return (level_sum - explore_thresh) / scale_factor + explore_thresh

    def is_trainer_battle_now(self) -> bool:
        return self.read_m(0xD119) > 0

    def step(self, action):
        self.step_count += 1
        self.check_manual_control()
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)
        self.update_seen_coords(action)
        self.update_explore_map()
        self.update_heal_reward()
        self.update_new_map_reward()
        self.append_agent_stats(action)

        is_in_battle = self.read_m(0xD116) != 0 
        trainer_class = self.read_m(0xD119)
        is_trainer_battle = (trainer_class > 0)
        
        mult = self.get_cap_multiplier(is_trainer_battle)

        # 전투 보상
        ENTRY_REWARD = 0.05
        FLEE_PENALTY = -0.25
        KO_REWARD    = 1.00

        if (is_in_battle and not self.prev_in_battle):
            self.total_battle_bonus += ENTRY_REWARD * mult
            self.battle_last_enemy_hp = self.read_hp(0xD0FF)

        if is_in_battle:
            self.battle_last_enemy_hp = self.read_hp(0xD0FF)

        if (not is_in_battle and self.prev_in_battle):
            if self.battle_last_enemy_hp == 0:
                self.total_battle_bonus += KO_REWARD * mult
            else:
                self.total_battle_bonus += FLEE_PENALTY * mult

        self.prev_in_battle = is_in_battle

        # 경험치
        current_exp = self.read_party_exp()
        if current_exp > self.last_exp:
            exp_gain = current_exp - self.last_exp
            if exp_gain < 5000:
                self.total_exp_reward += (math.log(exp_gain + 1) * 2 + 0.2) * mult
        self.last_exp = current_exp
        
        # 데미지
        current_enemy_hp = self.read_hp(0xD0FF)
        enemy_max_hp = self.read_hp(0xD101)
        
        if is_in_battle and enemy_max_hp > 0:
            if current_enemy_hp < self.last_enemy_hp and self.last_enemy_hp <= enemy_max_hp:
                damage = self.last_enemy_hp - current_enemy_hp
                if 0 < damage <= enemy_max_hp:
                    self.total_dmg_reward += damage * mult 
        
        # 레벨
        current_level_sum = self.get_levels_sum()
        if current_level_sum > self.last_level_sum:
            old_score = self.compute_scaled_level(self.last_level_sum)
            new_score = self.compute_scaled_level(current_level_sum)
            gain = new_score - old_score
            if gain > 0:
                self.accumulated_level_reward += gain * mult 
            self.last_level_sum = current_level_sum

        # 도감 획득 시 탐험 리셋
        mem_val = self.read_m(0xD88E)
        is_pokedex_event_done = (mem_val >> 5) & 1
        
        if is_pokedex_event_done == 1 and not self.has_reset_exploration:
            if self.print_rewards:
                print(f"\n📢 [Step {self.step_count}] 도감 획득! 탐험 보상 초기화.")
            badge_count = self.get_badges()
            lead_level = self.get_lead_level()
            allowed_cap = 11 + (badge_count * 8)
            if lead_level > allowed_cap and not is_trainer_battle:
                current_explore_plus = 2.0
            else:
                current_explore_plus = 1.0
            current_explore_val = self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.05 * current_explore_plus
            self.exploration_offset += current_explore_val
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
            "reward_dmg": float(self.progress_reward['dmg']),
            "reward_stuck": float(self.progress_reward.get('stuck', 0)),
            "reward_battle": float(self.progress_reward.get("battle", 0.0)),
            "reward_new_map": float(self.progress_reward.get("new_map", 0.0)), 
        }
        if getattr(self, "_stuck_move_count", 0) > 140:
            print(f"💀 [Stuck Death] 벽 비비기 감지! 에피소드 강제 종료.")
            return obs, -5.0, False, True, info
        return obs, new_reward, False, step_limit_reached, info

    
    def run_action_on_emulator(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)

    def append_agent_stats(self, action):
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        map_id = self.get_verified_map_id()
        if map_id is None or map_id < 0:
            map_id = -1 
        levels = [self.read_m(a) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]]
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
                "coord_count": len(getattr(self, "seen_coords", {})),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "total_reward": self.total_reward,
            }
        )

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

    def update_new_map_reward(self):
        current_map = self.get_verified_map_id()
        if current_map is not None and current_map != -1:
            if current_map not in self.visited_maps:
                self.visited_maps.add(current_map)
                self.total_new_map_reward += 10.0
                if self.print_rewards:
                    print(f"🗺️ [New Map] ID {current_map} 발견! (+10 Reward)")

    def update_seen_coords(self, action):
        if self.read_m(0xD116) != 0:
            return

        x_pos, y_pos, _, _ = self.get_game_coords()
        map_id = self.get_verified_map_id()
        if map_id is None or map_id < 0:
            return

        coord_key = f"x:{x_pos} y:{y_pos} m:{map_id}"

        if self._stuck_last_key is not None:
            try:
                last_map_id = int(self._stuck_last_key.split("m:")[-1])
                if last_map_id != int(map_id):
                    self._stuck_last_key = coord_key
                    self._stuck_move_count = 0
                    return
            except Exception:
                self._stuck_last_key = coord_key
                self._stuck_move_count = 0
                return

        self.seen_coords[coord_key] = self.seen_coords.get(coord_key, 0) + 1

        a = int(action)
        is_move_action = (0 <= a <= 3)  

        if not is_move_action:
            self._stuck_last_key = coord_key
            self._stuck_move_count = 0
            return

        if coord_key == self._stuck_last_key:
            self._stuck_move_count += 1
        else:
            self._stuck_last_key = coord_key
            self._stuck_move_count = 1

        if self._stuck_move_count > 30:
            self.accumulated_stuck_penalty -= 0.02
            
            if self._stuck_move_count % 30 == 0:
                print(
                    f"⚠️ [Stuck 감지] 이동 시도 후 같은 타일 연속 {self._stuck_move_count}회 "
                    f"(좌표: {x_pos}, {y_pos}, map:{map_id}) -> 누적 페널티: {self.accumulated_stuck_penalty:.2f}",
                    end="\r"
                )

    def get_stuck_penalty(self):
        return self.accumulated_stuck_penalty

    def get_global_coords(self):
        x_pos, y_pos, map_group, map_number = self.get_game_coords()
        map_id = self.get_verified_map_id()
        if map_id is None or map_id < 0:
            return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
        try:
            return local_to_global(y_pos, x_pos, map_id)
        except Exception:
            return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2

    def update_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]: pass
        else: self.explore_map[c[0], c[1]] = 255

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
        new_total = sum([val for _, val in self.progress_reward.items()])
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
        return [int(bit) for i in range(event_flags_start, event_flags_end) for bit in f"{self.read_m(i):08b}"]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 5 
        poke_levels = [max(self.read_m(a) - min_poke_level, 0) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_badges(self):
        return self.bit_count(self.read_m(0xD57C))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xDA23, 0xDA24, 0xDA25, 0xDA26, 0xDA27, 0xDA28]]

    def read_party_exp(self):
        h = self.read_m(0xDA32)
        m = self.read_m(0xDA33)
        l = self.read_m(0xDA34)
        return (h << 16) | (m << 8) | l

    def get_all_events_reward(self):
        return max(sum([self.bit_count(self.read_m(i)) for i in range(event_flags_start, event_flags_end)]) - self.base_event_flags, 0)

    def get_game_state_reward(self, print_stats=False):
        current_exp_score = self.reward_scale * self.total_exp_reward * 1.0
        current_dmg_score = self.reward_scale * self.total_dmg_reward * 0.5
        current_level_score = self.reward_scale * self.accumulated_level_reward * 4.0
        current_heal_score = self.reward_scale * self.total_healing_rew * 2
        badge_count = self.get_badges()
        level_sum = self.get_levels_sum()
        battle_type = self.read_m(0xD116)
        is_trainer_battle = self.is_trainer_battle_now()
        lead_level = self.get_lead_level()
        allowed_level_cap = 9 + (badge_count * 8)

        explore_plus = 1.0
        if lead_level > allowed_level_cap and not is_trainer_battle:
            explore_plus = 3.0

        final_exp = current_exp_score
        final_dmg = current_dmg_score
        final_level = current_level_score
        final_heal = current_heal_score 

        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 5.0,
            "level": final_level, 
            "heal": final_heal,
            "exp": final_exp,
            "dead": self.reward_scale * self.died_count * -1.5,
            "badge": self.reward_scale * self.get_badges() * 20,
            "explore": self.exploration_offset + self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.05 * explore_plus,
            
            "stuck": self.reward_scale * self.accumulated_stuck_penalty * 1.0,
            
            "dmg": final_dmg,
            "battle": self.reward_scale * self.total_battle_bonus,
            "new_map": self.reward_scale * self.total_new_map_reward, 
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
        hp_sum = sum([self.read_hp(add) for add in [0xDA4C, 0xDA7C, 0xDAAC, 0xDADC, 0xDB0C, 0xDB3C]])
        max_hp_sum = sum([self.read_hp(add) for add in [0xDA4E, 0xDA7E, 0xDAAE, 0xDADE, 0xDB0E, 0xDB3E]])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    def bit_count(self, bits):
        return bin(bits).count("1")

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        verified_map_id = self.get_verified_map_id()
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(verified_map_id))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys(): return self.essential_map_locations[map_idx]
        else: return -1
        
    def check_manual_control(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_path, "agent_enabled.txt")
        
        if self.step_count == 0: 
            print(f"📂 [DEBUG] 파일 찾는 위치: {file_path}", flush=True)
            
        try:
            if not os.path.exists(file_path): return
            
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    content = f.read().strip().lower()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding='cp949') as f:
                    content = f.read().strip().lower()
                
            if content == "no":
                print(f"🛑 [Manual Mode] 감지됨!", flush=True)
                while True:
                    self.pyboy.tick()
                    try:
                        try:
                            with open(file_path, "r", encoding='utf-8') as f:
                                new_content = f.read().strip().lower()
                        except UnicodeDecodeError:
                            with open(file_path, "r", encoding='cp949') as f:
                                new_content = f.read().strip().lower()
                                
                        if new_content != "no":
                            print(f"▶️ [Auto Mode] 재개!", flush=True)
                            break 
                    except: pass
        except Exception as e: pass
