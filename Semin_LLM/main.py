import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# 사용자 모듈 임포트
import config
import utils
from GoldEnv import GoldEnv
import callbacks


def make_env(rank, env_conf, ai_conf):
    """
    멀티프로세싱 환경 생성 공장 함수
    """

    def _init():
        full_conf = env_conf.copy()
        full_conf["ai_config"] = ai_conf
        full_conf["reward_weights"] = config.REWARD_WEIGHTS

        env = GoldEnv(full_conf)
        env.reset(seed=rank)
        return env

    return _init


if __name__ == "__main__":
    print("🚀 [초기화] 학습 설정을 불러오는 중...")

    # ==========================================
    # 1. 세션(로그) 폴더 자동 생성
    # ==========================================
    session_idx = utils.get_next_index(config.SESSIONS_DIR, "session")
    current_session_name = f"session_{session_idx}"
    current_session_path = os.path.join(config.SESSIONS_DIR, current_session_name)

    tensorboard_log_dir = os.path.join(current_session_path, "tensorboard_logs")
    checkpoint_dir = os.path.join(current_session_path, "checkpoints")

    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"📂 [저장소] 이번 학습 로그는 '{current_session_name}'에 저장됩니다.")

    # ==========================================
    # 2. 게임 환경 생성
    # ==========================================
    num_cpu = config.train_config["n_envs"]
    print(f"🎮 [환경] {num_cpu}개의 코어로 학습을 준비합니다.")

    env = SubprocVecEnv([
        make_env(i, config.env_config, config.ai_config)
        for i in range(num_cpu)
    ])

    # ==========================================
    # 3. 모델 로드 (커리큘럼 이어하기)
    # ==========================================
    best_model_path, start_badge = utils.get_best_badge_model(config.MODELS_DIR)

    if best_model_path:
        print(f"🔄 [이어하기] 배지 {start_badge}개 상태의 모델을 불러옵니다: {best_model_path}")
        try:
            model = PPO.load(
                best_model_path,
                env=env,
                tensorboard_log=tensorboard_log_dir,
                print_system_info=True
            )
            print(f"✅ 모델 로드 완료! (배지 {start_badge}개 맞춤형 보상 체계 적용)")
        except Exception as e:
            print(f"⚠️ 모델 로드 실패 ({e}). 새로 시작합니다.")
            model = PPO(
                "CnnPolicy", env, verbose=1,
                tensorboard_log=tensorboard_log_dir,
                learning_rate=config.train_config["learning_rate"],
                n_steps=config.train_config["n_steps"],
                batch_size=config.train_config["batch_size"]
            )
    else:
        print("✨ [새로하기] 저장된 배지 모델이 없습니다. 0부터 시작합니다.")
        model = PPO(
            "CnnPolicy", env, verbose=1,
            tensorboard_log=tensorboard_log_dir,
            learning_rate=config.train_config["learning_rate"],
            n_steps=config.train_config["n_steps"],
            batch_size=config.train_config["batch_size"]
        )

    # ==========================================
    # 4. 학습 시작
    # ==========================================
    my_callbacks = callbacks.get_callbacks(checkpoint_dir)

    print(f"🔥 [시작] 총 {config.train_config['total_timesteps']} 스텝 학습을 시작합니다!")

    try:
        model.learn(
            total_timesteps=config.train_config["total_timesteps"],
            callback=my_callbacks,
            progress_bar=True,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\n🛑 [중단] 사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n❌ [에러] 학습 중 문제가 발생했습니다: {e}")
    finally:
        env.close()
        final_save_path = os.path.join(current_session_path, "final_model_session_end")
        model.save(final_save_path)
        print("💾 [종료] 세션 종료 모델이 저장되었습니다.")