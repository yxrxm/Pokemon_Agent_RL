import uuid
import json
import os
from pathlib import Path

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import mediapy as media
from einops import repeat
from gymnasium import Env, spaces
from groq import Groq


from global_map import local_to_global, GLOBAL_MAP_SHAPE

event_flags_start = 0xD7B7
event_flags_end = 0xD8B6


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
        self.explore_weight = 1 if "explore_weight" not in config else config["explore_weight"]
        self.reward_scale = 1 if "reward_scale" not in config else config["reward_scale"]
        self.instance_id = str(uuid.uuid4())[:8] if "instance_id" not in config else config["instance_id"]

        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        # 성도 지방 주요 맵 ID
        self.essential_map_locations = {
            v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        }

        # =========================================================================
        # [LLM 통합] 1. Groq (Llama 3) 설정 및 초기화
        # =========================================================================
        # 아까 발급받은 gsk_로 시작하는 키를 여기에 넣으세요!
        self.groq_api_key = "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"

        try:
            self.llm_client = Groq(api_key=self.groq_api_key)
            self.use_llm_reward = True
            print("✅ [성공] Groq (Llama 3) 연결 완료: Reward Shaper 활성화됨")
        except Exception as e:
            print(f"❌ [에러] Groq 연결 오류: {e}")
            self.use_llm_reward = False

        self.llm_update_freq = 2048  # LLM 호출 주기 (스텝)
        self.last_llm_reason = "Initial State"

        # 동적 보상 가중치 (기본값)
        self.reward_weights = {
            "event": 4.0, "heal": 10.0, "badge": 10.0,
            "explore": 0.1, "stuck": -0.05, "level": 1.0
        }
        # =========================================================================

        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN, WindowEvent.PRESS_ARROW_LEFT, WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_BUTTON_B, WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN, WindowEvent.RELEASE_ARROW_LEFT, WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A, WindowEvent.RELEASE_BUTTON_B, WindowEvent.RELEASE_BUTTON_START
        ]

        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks)
        self.coords_pad = 12
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.enc_freqs = 8

        self.observation_space = spaces.Dict({
            "screens": spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8),
            "health": spaces.Box(low=0, high=1),
            "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,)),
            "badges": spaces.MultiBinary(8),
            "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
            "map": spaces.Box(low=0, high=255, shape=(self.coords_pad * 4, self.coords_pad * 4, 1), dtype=np.uint8),
            "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)] * self.frame_stacks)
        })

        head = "null" if config["headless"] else "SDL2"
        self.pyboy = PyBoy(config["gb_path"], window=head)
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

        self.base_event_flags = sum([
            self.bit_count(self.read_m(i)) for i in range(event_flags_start, event_flags_end)
        ])
        self.current_event_flags_set = {}
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]
        if reduce_res:
            game_pixels_render = (downscale_local_mean(game_pixels_render, (2, 2, 1))).astype(np.uint8)
        return game_pixels_render

    def _get_obs(self):
        screen = self.render()
        self.update_recent_screens(screen)
        level_sum = 0.02 * sum([self.read_m(a) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]])

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

    # =========================================================================
    # [수정] 겁쟁이 방지용 프롬프트 (침대 탈출 버전)
    # =========================================================================
    def update_reward_weights_by_gemini(self):
        if not self.agent_stats or not self.use_llm_reward:
            return

        current_stat = self.agent_stats[-1]

        # 1. 시스템 프롬프트
        system_msg = "You are a bold and aggressive AI coach for Pokemon Gold. You must output ONLY valid JSON."

        # 2. 사용자 프롬프트 (강력한 탐험 유도)
        user_msg = f"""
        Current Status:
        - Map ID: {current_stat['map']} (Coords: {current_stat['x']}, {current_stat['y']})
        - Level Sum: {current_stat['levels_sum']}
        - Badges: {self.get_badges()}
        - Explored: {len(self.seen_coords)}
        - Deaths: {self.died_count}

        Analyze and return reward weights.

        CRITICAL RULES (Follow these strictly):
        1. "explore" MUST ALWAYS be higher than "heal". (Exploration is the main goal).
        2. Do NOT boost "heal" above 3.0 even if dying often. (Too high heal reward causes looping at Pokemon Center).
        3. If stuck (no new explored spots), boost "explore" to 8.0+.
        4. "event" and "badge" are high priority rewards (5.0+).

        Strategy:
        - Don't be afraid of death. Focus on PROGRESS.
        - If Badges == 0: Prioritize "explore" (5.0) and "level" (3.0). "heal" should be low (0.5).

        Example JSON output:
        {{
            "event": 5.0, "heal": 0.5, "badge": 10.0, 
            "explore": 6.0, "level": 2.0, "stuck": -0.1,
            "reason": "Push forward! Don't stay at home."
        }}
        """

        try:
            # Groq 호출
            completion = self.llm_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,  # 창의성 약간 높임 (다양한 시도 유도)
                response_format={"type": "json_object"}
            )

            # 응답 파싱
            response_text = completion.choices[0].message.content
            new_weights = json.loads(response_text)

            # 가중치 업데이트
            for k in self.reward_weights.keys():
                if k in new_weights:
                    self.reward_weights[k] = float(new_weights[k])

            self.last_llm_reason = new_weights.get("reason", "No reason provided")

            # 콘솔 출력
            if self.print_rewards:
                print(f"\n🦙 [Groq] {self.last_llm_reason}")
                print(f"   Weights: {self.reward_weights}")

            # 파일로 기록
            log_filename = self.s_path / f"llm_strategy_log_{self.instance_id}.txt"
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"=== Step: {self.step_count} ===\n")
                f.write(f"📍 상태: 맵{current_stat['map']} / 데스:{self.died_count}\n")
                f.write(f"🤖 이유: {self.last_llm_reason}\n")
                f.write(f"⚖️ 결과: {json.dumps(self.reward_weights, ensure_ascii=False)}\n")
                f.write("\n")

        except Exception as e:
            print(f"\n⚠️ [Groq Error] {e}")

    # =========================================================================
    # [LLM 통합] 3. step 함수 수정 (트리거 추가)
    # =========================================================================
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

        # [트리거] LLM 호출
        if self.step_count > 0 and self.step_count % self.llm_update_freq == 0:
            self.update_reward_weights_by_gemini()

        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        step_limit_reached = self.check_if_done()
        obs = self._get_obs()

        if self.step_count % 100 == 0:
            # Event Flag Update Logic (Simple Check)
            pass

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
        levels = [self.read_m(a) for a in [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]]
        self.agent_stats.append({
            "step": self.step_count, "x": x_pos, "y": y_pos, "map": map_n,
            "max_map_progress": self.max_map_progress, "last_action": action,
            "pcount": self.read_m(0xDA22), "levels": levels, "levels_sum": sum(levels),
            "ptypes": self.read_party(), "hp": self.read_hp_fraction(),
            "coord_count": len(self.seen_coords), "deaths": self.died_count,
            "badge": self.get_badges(), "event": self.progress_reward["event"],
            "healr": self.total_healing_rew,
        })

    def start_video(self):
        if self.full_frame_writer is not None: self.full_frame_writer.close()
        if self.model_frame_writer is not None: self.model_frame_writer.close()
        if self.map_frame_writer is not None: self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f"full_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        model_name = Path(f"model_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60, input_format="gray")
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_shape[:2], fps=60,
                                                    input_format="gray")
        self.model_frame_writer.__enter__()
        map_name = Path(f"map_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(base_dir / map_name, (self.coords_pad * 4, self.coords_pad * 4),
                                                  fps=60, input_format="gray")
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])
        self.map_frame_writer.add_image(self.get_explore_map())

    def get_game_coords(self):
        return (self.read_m(0xD20D), self.read_m(0xD20E), self.read_m(0xDA01))

    def update_seen_coords(self):
        if self.read_m(0xD116) == 0:
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
            pass
        else:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0] - self.coords_pad:c[0] + self.coords_pad, c[1] - self.coords_pad:c[1] + self.coords_pad]
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
        return self.step_count >= self.max_steps - 1

    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [int(bit) for i in range(event_flags_start, event_flags_end) for bit in f"{self.read_m(i):08b}"]

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [max(self.read_m(a) - min_poke_level, 0) for a in
                       [0xDA49, 0xDA79, 0xDAA9, 0xDAD9, 0xDB09, 0xDB39]]
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

    def get_badges(self):
        return self.bit_count(self.read_m(0xD57C))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xDA23, 0xDA24, 0xDA25, 0xDA26, 0xDA27, 0xDA28]]

    def get_all_events_reward(self):
        return max(sum([self.bit_count(self.read_m(i)) for i in
                        range(event_flags_start, event_flags_end)]) - self.base_event_flags, 0)

    # =========================================================================
    # [LLM 통합] 4. 보상 계산식 (가중치 적용)
    # =========================================================================
    def get_game_state_reward(self, print_stats=False):
        state_scores = {
            "event": self.reward_scale * self.update_max_event_rew() * self.reward_weights["event"],
            "heal": self.reward_scale * self.total_healing_rew * self.reward_weights["heal"],
            "badge": self.reward_scale * self.get_badges() * self.reward_weights["badge"],
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * self.reward_weights["explore"],
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * self.reward_weights["stuck"],
            "level": self.reward_scale * self.get_levels_reward() * self.reward_weights["level"]
        }
        return state_scores

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
        map_idx = self.read_m(0xDA01)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1