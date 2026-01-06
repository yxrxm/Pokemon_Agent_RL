from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from pathlib import Path

from GoldEnv import GoldEnv

def make_env():
    env_config = {
        "headless": False,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": "./init.state",
        "max_steps": 2_000_000,
        "print_rewards": False,
        "save_video": False,
        "gb_path": "./PokeGold.gbc",
        "debug": False,
        "session_path": Path("ppo_world_session"),
        "fast_video": True,
    }
    return GoldEnv(env_config)

if __name__ == "__main__":
    set_random_seed(0)

    env = DummyVecEnv([make_env])

    PRETRAIN_ZIP = "poke_26214400.zip"  # 👈 네가 말한 전이학습 모델

    if Path(PRETRAIN_ZIP).exists():
        print(f"[INFO] Loading pretrained PPO: {PRETRAIN_ZIP}")
        model = PPO.load(PRETRAIN_ZIP, env=env)
    else:
        print("[WARN] Pretrained PPO not found, training from scratch")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
        )

    model.learn(total_timesteps=2_000_000)

    model.save("world_ppo.zip")
    env.close()
