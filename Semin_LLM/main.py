import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder #화면 껐을 때 녹화용
# 사용자 모듈 임포트
import config
import utils
from GoldEnv import GoldEnv
import callbacks


def make_env(rank, env_conf, ai_conf):
    """
    멀티프로세싱 환경 생성 공장 함수
    """
    #config.py에 있는 설정값들을 받아옴.
    def _init():
        configs = env_conf.copy()
        configs["ai_config"] = ai_conf
        configs["reward_weights"] = config.REWARD_WEIGHTS
        
        #GoldEnv Class 생성
        env = GoldEnv(configs)
        #reset
        env.reset(seed=rank)
        return env

    return _init


if __name__ == "__main__":
    SEED = config.train_config.get("seed", 42)
    set_random_seed(SEED)
    print("학습 설정 불러오는 중")

    #여러 정보를 저장할 session file
    session_idx = utils.get_next_index(config.SESSIONS_DIR, "session")
    current_session_name = f"session_{session_idx}"
    current_session_path = os.path.join(config.SESSIONS_DIR, current_session_name)

    tensorboard_log_dir = os.path.join(current_session_path, "tensorboard_logs")

    #tensorboard --logdir=지정한_폴더_경로
    checkpoint_dir = os.path.join(current_session_path, "checkpoints")

    #기본 디렉토리 생성
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"파일 저장 위치: {current_session_name}")

    #cpu 개수 받아오기
    num_cpu = config.train_config["n_envs"]
    print(f"{num_cpu}개의 코어로 학습을 준비합니다.")

    env = SubprocVecEnv([
        make_env(i, config.env_config, config.ai_config)
        for i in range(num_cpu)
    ])

    # save_video
    if config.env_config.get("save_video", False):
        print("🎥 영상 녹화 기능이 활성화되었습니다.")

        video_folder = os.path.join(current_session_path, "videos")
        os.makedirs(video_folder, exist_ok=True)

        # 영상 길이 (스텝 수)
        video_length = 3000  # 약 1~2분 분량

        # 녹화 빈도 (몇 스텝마다 녹화할지)
        record_freq = 50_000

        env = VecVideoRecorder(
            env,
            video_folder,
            # x는 현재 총 스텝 수. 0이거나 record_freq 배수일 때 녹화 시작
            record_video_trigger=lambda x: x == 0 or x % record_freq == 0,
            video_length=video_length,
            name_prefix="agent-video"
        )

    #3. 모델 로드(커리큘럼 이어하기 로직)
    best_model_path, start_badge = utils.get_best_badge_model(config.MODELS_DIR)

    if best_model_path:
        #뱃지 갯수에 따른 모델 불러오기
        print(f" [이어하기] 배지 {start_badge}개 상태의 모델을 불러옵니다: {best_model_path}")
        try:
            model = PPO.load(
                best_model_path,
                env=env,
                tensorboard_log=tensorboard_log_dir,
                print_system_info=True
            )
            print(f"모델 로드 성공! (배지 {start_badge}개 맞춤형 보상 체계 적용)")
        except Exception as e:
            print(f"모델 로드 실패: {e}, 새로 시작.")
            model = PPO(
                "CnnPolicy", env, verbose=1,
                tensorboard_log=tensorboard_log_dir,
                learning_rate=config.train_config["learning_rate"],
                n_steps=config.train_config["n_steps"],
                batch_size=config.train_config["batch_size"]
            )
    else:
        print("저장된 배지 모델이 없습니다. 0부터 시작합니다.")
        model = PPO(
            "CnnPolicy", env, verbose=1,
            tensorboard_log=tensorboard_log_dir,
            learning_rate=config.train_config["learning_rate"],
            n_steps=config.train_config["n_steps"],
            batch_size=config.train_config["batch_size"]
        )

    #4. 학습 본격 시작
    my_callbacks = callbacks.get_callbacks(checkpoint_dir)
    total_timesteps = config.train_config['total_timesteps']

    print(f"설정 스텝: {total_timesteps} // 학습을 시작합니다!")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=my_callbacks,
            progress_bar=True,
            reset_num_timesteps=False #timesteps 초기화
        )
    except KeyboardInterrupt:
        print("\n사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n학습 중 문제가 발생했습니다: {e}")
    finally:
        #안전한 종료 및 저장
        env.close()
        final_save_path = os.path.join(current_session_path, "final_model_session_end")
        model.save(final_save_path)
        print(f"세션 종료 모델이 저장되었습니다. {final_save_path}")