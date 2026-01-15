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

from global_map import local_to_global, GLOBAL_MAP_SHAPE

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
                    with open(filename, "r", encoding="utf-8", errors='ignore') as f:
                        return json.load(f)
            except FileNotFoundError:
                return {} 

        # =========================================================================
        # 🗺️ [Hybrid Mapping] 맵은 크리스탈 구조 (Group 10=도라지)
        # =========================================================================

        # [Milestone] 주요 거점 보상 (50점)
        self.milestone_maps = {
            24003: 50.0,  # [User] 29번 도로
            26003: 50.0,  # [User] 무궁시티
            26002: 50.0,  # [User] 31번 도로
            26001: 50.0,  # [User] 30번 도로
            10005: 50.0,  # [User] 도라지시티
            10007: 50.0,  # [User] 도라지 체육관
            5003: 50.0,   # [Crystal Logic] 고동마을
            11003: 50.0,  # [Crystal Logic] 금빛시티
        }

        # [Gym] 체육관 ID (1000점)
        self.gym_map_ids = {
            10007,  # [User] 도라지 체육관
            5004,   # [Crystal] 고동 체육관
            11004,  # [Crystal] 금빛 체육관
        }

        # 🔥 [핵심] 위치 인덱싱
        self.essential_map_locations = {
            # [Index 1] 연두마을
            24004: 1, 24001: 1, 24002: 1, 24005: 1, 24006: 1,

            # [Index 2] 29번 도로
            24003: 2, 26004: 2,

            # [Index 3] 무궁시티
            26003: 3, 26005: 3, 26006: 3, 26007: 3, 26008: 3,

            # [Index 4] 30번 도로
            26001: 4, 26009: 4,

            # [Index 5] 31번 도로
            26002: 5, 26010: 5, 26011: 5, 
            3070: 5, # 어둠의 동굴

            # [Index 6] 도라지시티 (Violet City)
            10005: 6, # 도라지시티
            10007: 6, # 체육관
            10010: 6, # 포켓몬 센터
            10006: 6, 10008: 6,
            3001: 6, 3002: 6, 3003: 6, # 모다피 탑

            # [Index 7] 32번 도로
            10001: 7, 10002: 7, 10009: 7,
            7000: 7, 7001: 7, 7002: 7, # 알프의 유적

            # [Index 8] 연결동굴
            9000: 8, 9001: 8, 9002: 8,

            # [Index 9] 고동마을 (Azalea Town)
            5003: 9, 5004: 9, 5001: 9, 5002: 9, 5005: 9, 5006: 9,
            13000: 9, 13001: 9, # 야돈 우물
            5000: 9, # 33번 도로

            # [Index 10] 금빛시티 (Goldenrod City)
            11003: 10, 11004: 10, 11002: 10, 11005: 10, 11000: 10,
            11006: 10, 11007: 10,
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
        
        # 🟢 [Battle: Gold] 초기 적 HP 읽기 (Gold 주소: 0xD0FF)
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
        if diff == 2: return 1.0
        if diff == 1: return 0.7
        return 0.4

    def compute_scaled_level(self, level_sum):
        explore_thresh = 22
        scale_factor = 4
        if level_sum < explore_thresh:
            return level_sum
        else:
            return (level_sum - explore_thresh) / scale_factor + explore_thresh

    def is_trainer_battle_now(self) -> bool:
        # 🟢 [Battle: Gold] 트레이너 클래스 (0xD119)
        return self.read_m(0xD119) > 0

    def step(self, action):
        self.step_count += 1
        
        # 🔥 Panic Mode
        if getattr(self, "_stuck_move_count", 0) > 30:
            if np.random.rand() < 0.5:
                action = self.action_space.sample()

        self.check_manual_control()
        self.run_action_on_emulator(action)
        self.update_recent_actions(action)
        self.update_seen_coords(action)
        self.update_explore_map()
        self.update_heal_reward()
        self.update_new_map_reward()
        self.append_agent_stats(action)

        # 🟢 [Battle: Gold] 전투 상태 (0xD116)
        battle_mem = self.read_m(0xD116)
        is_in_battle = battle_mem != 0 
        
        trainer_class = self.read_m(0xD119)
        is_trainer_battle = (trainer_class > 0)
        
        mult = self.get_cap_multiplier(is_trainer_battle)

        # === 전투 보상 ===
        ENTRY_REWARD = 0.5
        FLEE_PENALTY = 0.0
        KO_REWARD    = 10.00 if is_trainer_battle else 1.00 

        # 🟢 [Battle: Gold] 적 HP (0xD0FF), Max HP (0xD101)
        current_enemy_hp = self.read_hp(0xD0FF)
        enemy_max_hp = self.read_hp(0xD101)

        # 🛡️ [Reward Fix] Max HP가 0이면 보상 로직이 멈추는 문제 해결
        # 어떤 경우에도 enemy_max_hp가 유효한 값을 가지도록 보정
        if is_in_battle and enemy_max_hp == 0:
            enemy_max_hp = max(self.last_enemy_hp, current_enemy_hp, 1)

        if (is_in_battle and not self.prev_in_battle):
            self.total_battle_bonus += ENTRY_REWARD * mult
            self.battle_last_enemy_hp = current_enemy_hp

        if is_in_battle:
            self.battle_last_enemy_hp = current_enemy_hp
            self.total_battle_bonus -= 0.005

        if (not is_in_battle and self.prev_in_battle):
            if self.battle_last_enemy_hp == 0:
                self.total_battle_bonus += KO_REWARD * mult
                if is_trainer_battle and self.print_rewards:
                    print(f"🥊 [Battle] 트레이너 격파! (+{KO_REWARD})")
            else:
                self.total_battle_bonus += FLEE_PENALTY * mult

        self.prev_in_battle = is_in_battle

        current_exp = self.read_party_exp()
        if current_exp > self.last_exp:
            exp_gain = current_exp - self.last_exp
            if exp_gain < 5000:
                self.total_exp_reward += (exp_gain * 0.15) * mult
        self.last_exp = current_exp
        
        # 데미지 보상 로직 (보정된 Max HP 덕분에 정상 작동)
        if is_in_battle and enemy_max_hp > 0:
            if current_enemy_hp < self.last_enemy_hp and self.last_enemy_hp <= enemy_max_hp:
                damage = self.last_enemy_hp - current_enemy_hp
                if 0 < damage <= enemy_max_hp:
                    self.total_dmg_reward += damage * mult 
        
        current_level_sum = self.get_levels_sum()
        if current_level_sum > self.last_level_sum:
            old_score = self.compute_scaled_level(self.last_level_sum)
            new_score = self.compute_scaled_level(current_level_sum)
            gain = new_score - old_score
            if gain > 0:
                self.accumulated_level_reward += gain * mult 
            self.last_level_sum = current_level_sum

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
            "stats_steps": float(self.step_count),
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

        # Stuck Death (500회)
        if getattr(self, "_stuck_move_count", 0) > 500:
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
        return (self.read_m(0xD20E), self.read_m(0xD20D), self.read_m(0xDA00), self.read_m(0xDA01))

    # 🔥 [핵심] 수동 체크와 ID 동일화 (Group * 1000 + Number)
    # global_map 의존성을 제거하고 직접 계산하여 ID 불일치 문제 해결
    def get_verified_map_id(self):
        _, _, map_group, map_n = self.get_game_coords()
        # [수정] 수동 체크와 동일한 공식 적용
        current_raw_id = (map_group * 1000) + map_n

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
                
                # 1. 체육관 (1000점)
                if current_map in self.gym_map_ids:
                    self.total_new_map_reward += 1000.0
                    if self.print_rewards:
                        print(f"🏟️ [Gym] 체육관 발견! 인생 역전! (+1000 Reward)")
                
                # 2. Milestone (50점)
                elif current_map in self.milestone_maps:
                    bonus = self.milestone_maps[current_map]
                    self.total_new_map_reward += bonus
                    if self.print_rewards:
                        print(f"🚩 [Milestone] 주요 거점(ID:{current_map}) 돌파! (+{bonus} Reward)")
                
                # 3. 일반 맵 (10점)
                else:
                    self.total_new_map_reward += 10.0
                    if self.print_rewards:
                        print(f"🗺️ [New Map] ID {current_map} 발견! (+10 Reward)")

    def update_seen_coords(self, action):
        # 🟢 [Battle: Gold] 전투 상태 (0xD116)
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
        
        # A(4), B(5) 버튼은 Stuck 카운트 제외 (방향키만 체크)
        if action < 4: 
            if coord_key == self._stuck_last_key:
                self._stuck_move_count += 1
            else:
                self._stuck_last_key = coord_key
                self._stuck_move_count = 0
        else:
            pass

        if self._stuck_move_count > 30:
            self.accumulated_stuck_penalty -= 0.03
            
            if self._stuck_move_count % 30 == 0:
                print(
                    f"⚠️ [Stuck] {self._stuck_move_count} steps (x:{x_pos}, y:{y_pos}, m:{map_id}) -> Penalty: {self.accumulated_stuck_penalty:.2f}",
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
        current_exp_score = self.reward_scale * self.total_exp_reward * 1.4
        current_dmg_score = self.reward_scale * self.total_dmg_reward * 0.4
        current_level_score = self.reward_scale * self.accumulated_level_reward * 10.0
        current_heal_score = self.reward_scale * self.total_healing_rew * 2
        badge_count = self.get_badges()
        level_sum = self.get_levels_sum()
        
        # 🟢 [Battle: Gold] 전투 상태 (0xD116)
        is_in_battle = self.read_m(0xD116) != 0
        is_trainer_battle = self.is_trainer_battle_now()
        lead_level = self.get_lead_level()
        allowed_level_cap = 11 + (badge_count * 8)

        explore_plus = 1.0
        if lead_level > allowed_level_cap and not is_trainer_battle:
            explore_plus = 3.0

        final_exp = current_exp_score
        final_dmg = current_dmg_score
        final_level = current_level_score
        final_heal = current_heal_score 

        # Calculate Weight
        use_prog = self.max_map_progress
        map_id = self.get_verified_map_id()
        curr_prog = self.get_map_progress(map_id)
        
        if curr_prog != -1:
            use_prog_for_weight = curr_prog
        else:
            use_prog_for_weight = self.max_map_progress

        prog_weight = 1.0 + (use_prog_for_weight * 0.2)

        # Cut-off Rule: 2단계 이상 후퇴 시 탐험 점수 0점
        base_explore = len(self.seen_coords) * 0.2
        
        if curr_prog != -1:
            if (self.max_map_progress - curr_prog) >= 2:
                base_explore = 0.0
        
        explore_score = self.exploration_offset + (self.reward_scale * self.explore_weight * base_explore * explore_plus * prog_weight)

        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * 10.0,
            "level": current_level_score, 
            "heal": final_heal,
            "exp": final_exp * prog_weight,
            "dead": self.reward_scale * self.died_count * -1.0,
            "badge": self.reward_scale * self.get_badges() * 50,
            "explore": explore_score,
            "stuck": self.reward_scale * self.accumulated_stuck_penalty * 1.0,
            "dmg": final_dmg * prog_weight,
            "battle": self.reward_scale * self.total_battle_bonus,
            "new_map": self.reward_scale * self.total_new_map_reward, 
        }
        return state_scores

    def update_max_op_level(self):
        # 🟢 [Battle: Gold] 적 레벨 (0xD0FC)
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
            print(f"📂 [DEBUG] Path: {file_path}", flush=True)
            
        try:
            if not os.path.exists(file_path): return
            
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    content = f.read().strip().lower()
            except UnicodeDecodeError:
                with open(file_path, "r", encoding='cp949') as f:
                    content = f.read().strip().lower()
                
            if content == "no":
                print(f"🛑 [Manual Mode]", flush=True)
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
                            print(f"▶️ [Auto Mode]", flush=True)
                            break 
                    except: pass
        except Exception as e: pass