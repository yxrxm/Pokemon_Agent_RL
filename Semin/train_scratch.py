import os
import re
import glob
from pathlib import Path
from GoldEnv import GoldEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback


# === [유틸리티] 다음 세션 번호 구하기 (1, 2, 3...) ===
def get_next_session_id(folder_path, prefix="final_model_"):
    if not os.path.exists(folder_path):
        return 1

    files = glob.glob(os.path.join(folder_path, f"{prefix}*.zip"))

    max_id = 0
    for f in files:
        filename = os.path.basename(f)
        match = re.search(rf"{prefix}(\d+)", filename)
        if match:
            number = int(match.group(1))
            if number > max_id:
                max_id = number

    return max_id + 1


# === 1. 환경 생성 함수 ===
def make_env(rank, env_conf, seed=0):
    def _init():
        env = GoldEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    # 저장 폴더 설정
    log_dir = "./runs"
    save_dir = "./Data_saver"
    session_root_dir = "./sessions"  # [추가] 세션들을 모아둘 부모 폴더

    Path(log_dir).mkdir(exist_ok=True)
    Path(save_dir).mkdir(exist_ok=True)
    Path(session_root_dir).mkdir(exist_ok=True)  # [추가] sessions 폴더 생성

    # 1. 세션 번호 계산 (1, 2, 3...)
    sess_num = get_next_session_id(save_dir, prefix="final_model_")
    sess_id = str(sess_num)

    # [수정] 세션 폴더 경로를 'sessions/session_1' 형태로 변경
    sess_path = Path(session_root_dir) / f'session_{sess_id}'
    sess_path.mkdir(exist_ok=True)

    print(f"=== 학습 시작! (세션 번호: {sess_id}) ===")
    print(f"=== 세션 저장 경로: {sess_path} ===")

    # === 2. 환경 설정 ===
    env_config = {
        'headless': False,  # [주의] 멀티프로세싱(num_cpu > 1) 사용 시 True 권장!
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': './init.state',
        'max_steps': 2048 * 50,
        'print_rewards': True,
        'save_video': False,  # [팁] 용량이 걱정되면 False로 바꾸세요
        'fast_video': False,
        'session_path': sess_path,  # 수정된 경로 전달
        'gb_path': './PokeGold.gbc',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }

    num_cpu = 2 # 본인 CPU 코어 수에 맞게 조절 (headless=False일 땐 1 권장)
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # === 3. AI 모델 생성 (RecurrentPPO + LSTM) ===
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "enable_critic_lstm": False,
        }
    )

    print(f"모델 생성 완료! LSTM(기억력)이 장착되었습니다.")

    # === 4. 콜백(자동 저장) 설정 ===
    checkpoint_callback = CheckpointCallback(
        save_freq=500000 // num_cpu,
        save_path=save_dir,
        name_prefix=f'poke_gold_{sess_id}'
    )

    tensorboard_callback = TensorboardCallback(log_dir=log_dir)

    callbacks = CallbackList([checkpoint_callback, tensorboard_callback])

    # === 5. 학습 시작 ===
    print("학습 루프에 진입합니다... (중단하려면 Ctrl+C)")
    try:
        model.learn(total_timesteps=100_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("\n사용자에 의해 학습이 중단되었습니다.")

    # 최종 저장
    final_save_path = os.path.join(save_dir, f"final_model_{sess_id}")
    model.save(final_save_path)
    print(f"학습 종료 및 저장 완료: {final_save_path}.zip")

    env.close()