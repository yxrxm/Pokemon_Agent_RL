import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
from GoldEnv import GoldEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = GoldEnv(env_conf)
        # env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def get_most_recent_zip_with_age(folder_path):
    # Get all zip files in the folder
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))

    if not zip_files:
        return None, None  # Return None if no zip files are found

    # Find the most recently modified zip file
    most_recent_zip = max(zip_files, key=os.path.getmtime)

    # Calculate how old the file is in hours
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600  # Convert seconds to hours

    return most_recent_zip, age_in_hours


if __name__ == '__main__':

    sess_path = Path(f'session_run_only')
    ep_length = 2 ** 23

    env_config = {
        'headless': False, 'save_final_state': True, 'early_stop': False,
        'action_freq': 24, 'init_state': './init.state', 'max_steps': ep_length,
        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
        'gb_path': './PokeGold.gbc', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
    }

    #num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    #env = make_env(0, env_config)()  # SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    set_random_seed(0)
    env = DummyVecEnv([lambda: GoldEnv(env_config)])
    model = PPO("MultiInputPolicy", env)
    # # env_checker.check_env(env)
    # most_recent_checkpoint, time_since = get_most_recent_zip_with_age("runs")
    # if most_recent_checkpoint is not None:
    #     # file_name = most_recent_checkpoint 원본
    #     file_name = os.path.abspath(most_recent_checkpoint)
    #     print(f"using checkpoint: {file_name}, which is {time_since} hours old")
    #
    # # could optionally manually specify a checkpoint here
    # # file_name = "runs/poke_41943040_steps.zip"
    # print('\nloading checkpoint')
    # model = PPO.load(file_name, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})
    #
    # # keyboard.on_press_key("M", toggle_agent)
    obs= env.reset()
    while True:
        try:
            # agent_enabled.txt 파일을 매 프레임마다 확인합니다.
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except FileNotFoundError:
            agent_enabled = False
        except Exception as e:
            agent_enabled = False
            print(f"파일 읽기 오류: {e}")

        if agent_enabled:
            # AI가 활성화되면, (랜덤) 행동을 예측하여 수행합니다.
            action, _states = model.predict(obs, deterministic=False)
            # VecEnv는 4개의 값을 반환합니다.
            obs, rewards, dones, infos = env.step(action)

            # truncated 값은 infos 리스트의 0번째 딕셔너리에서 따로 추출해야 합니다.
            # (환경을 1개만 실행한다는 가정 하에 0번 인덱스 사용)
            info = infos[0]
            truncated = info.get('TimeLimit.truncated', False) or info.get('truncated', False)

            # ... (루프의 끝 부분)
            # 종료 조건도 dones[0] 또는 truncated를 사용해야 합니다.
            if dones[0] or truncated:
                break
        else:
            # AI가 비활성화되면, 유저가 직접 플레이하거나 관전합니다.
            # (PyBoy 2.6.0 버전에 맞게 수정)
            env.envs[0].pyboy.tick(1, True)  # 1프레임 진행
            obs = env.envs[0]._get_obs()  # 현재 관찰 값만 업데이트
            truncated = env.envs[0].step_count >= env.envs[0].max_steps - 1

        # env.render() # VecEnv는 render()를 이렇게 호출하지 않습니다.
        # headless=False이므로 창은 이미 떠 있습니다.

        if truncated:
            print("최대 스텝에 도달하여 에피소드를 종료합니다.")
            break

    print("실행 종료.")
env.close()

