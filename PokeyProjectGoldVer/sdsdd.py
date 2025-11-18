import os
import glob
import time
from os.path import exists
from pathlib import Path

from GoldEnv import GoldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

import subprocess
import sys
import webbrowser


LOG_DIR = Path(r"C:\tb_logs\gold_runs") #로그 저장 경로 (일단은 절대경로로 만듦)
TB_PORT = 6006


def make_env(rank, env_conf, seed=0):
    def _init():
        env = GoldEnv(env_conf)
        return env

    set_random_seed(seed)
    return _init


def get_most_recent_zip_with_age(folder_path):
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    if not zip_files:
        return None, None
    most_recent_zip = max(zip_files, key=os.path.getmtime)
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600
    return most_recent_zip, age_in_hours



#  TensorBoard / train_tb

def start_tensorboard(logdir: Path, port: int = 6006):
    """
    주어진 logdir에 대해 TensorBoard 서버를 백그라운드로 실행.
    """
    logdir = logdir.resolve()
    print(f"[sdsdd] TensorBoard starting... logdir={logdir}, port={port}")
    logdir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(logdir),
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    time.sleep(3)
    url = f"http://localhost:{port}"
    print(f"[sdsdd] TensorBoard running at {url}")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    return proc


def start_train_tb(checkpoint=None):
    print("[sdsdd] train_tb.py 실행 시작")

    if checkpoint:
        proc = subprocess.Popen(
            [sys.executable, "train_tb.py"],
            stdin=subprocess.PIPE,
        )
        proc.stdin.write((checkpoint + "\n").encode())
        proc.stdin.flush()
    else:
        proc = subprocess.Popen(
            [sys.executable, "train_tb.py"],
        )

    print(f"[sdsdd] train_tb.py PID={proc.pid}")
    return proc


def run_train_with_tensorboard():
    """
    1) train_tb.py 학습 시작
    2) 같은 logdir 기준으로 TensorBoard 실행
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    #train_tb 실행
    train_proc = start_train_tb(checkpoint=None)

    #텐서보드 실행
    tb_proc = start_tensorboard(LOG_DIR, port=TB_PORT)

    print("[sdsdd] RL 학습(train_tb) + TensorBoard 둘 다 실행 중")
    print("         학습 로그는 C:\\tb_logs\\gold_runs 기준으로 텐서보드에서 확인 가능")



#   기존 기능

def main_original():
    sess_path = Path("session_run_only")
    ep_length = 2 ** 23

    env_config = {
        "headless": False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "./init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "./PokeGold.gbc",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    set_random_seed(0)
    env = DummyVecEnv([lambda: GoldEnv(env_config)])
    model = PPO("MultiInputPolicy", env)

    obs = env.reset()
    while True:
        try:
            with open("agent_enabled.txt", "r", encoding="utf-8") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except FileNotFoundError:
            agent_enabled = False
        except Exception as e:
            agent_enabled = False
            print(f"파일 읽기 오류: {e}")

        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = env.step(action)

            info = infos[0]
            truncated = info.get("TimeLimit.truncated", False) or info.get("truncated", False)

            if dones[0] or truncated:
                break
        else:
            env.envs[0].pyboy.tick(1, True)
            obs = env.envs[0]._get_obs()
            truncated = env.envs[0].step_count >= env.envs[0].max_steps - 1

        if truncated:
            print("최대 스텝에 도달하여 에피소드를 종료합니다.")
            break

    print("실행 종료.")
    env.close()



#최종 main 진입점


def main():
    #RL 학습 + 텐서보드 백그라운드 실행
    run_train_with_tensorboard()

    #기존 기능
    main_original()


if __name__ == "__main__":
    main()
