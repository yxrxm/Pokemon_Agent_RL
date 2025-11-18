import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

from tensorboard_callback import TensorboardCallback
from GoldEnv import GoldEnv
from stream_agent_wrapper import StreamWrapper


def make_env(rank, env_conf, seed: int = 0):
    def _init():
        env = StreamWrapper(
            GoldEnv(env_conf),
            stream_metadata={
                "user": "gold-ppo",
                "env_id": rank,
                "color": "#447799",
                "extra": "",
            },
        )
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def main():
    tb_root = Path(r"C:\tb_logs\gold_runs") # 텐서보드 로그 저장 경로 (일단은 절대경로로 만듦)
    tb_root.mkdir(parents=True, exist_ok=True)

    # ===== 기본 설정 =====
    ep_length = 2048 * 80  # 163840

    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "./init.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": tb_root,
        "gb_path": "./PokeGold.gbc",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "extra_buttons": False,
    }

    print("[train_tb] env_config =")
    print(env_config)
    print("[train_tb] tensorboard/checkpoint root =", tb_root)

    num_cpu = 1
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2,
        save_path=tb_root,
        name_prefix="poke",
    )
    tb_callback = TensorboardCallback(tb_root)
    callbacks = CallbackList([checkpoint_callback, tb_callback])

    #PPO 설정
    train_steps_batch = ep_length // 64  # 2560
    print(f"[train_tb] train_steps_batch = {train_steps_batch}")

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=train_steps_batch,
        batch_size=512,
        n_epochs=1,
        gamma=0.997,
        ent_coef=0.01,
        tensorboard_log=None,  # TensorBoard 콜백에서 처리
    )

    print("[train_tb] model created")
    print(model.policy)

    total_timesteps = ep_length * num_cpu * 10000
    print(f"[train_tb] start learning, total_timesteps = {total_timesteps}")

    #학습 시작
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="poke_ppo_gold",
    )

    print("[train_tb] learn finished, exiting")


if __name__ == "__main__":
    main()
