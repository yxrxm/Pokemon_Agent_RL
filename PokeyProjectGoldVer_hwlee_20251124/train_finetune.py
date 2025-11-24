import os
import uuid
from pathlib import Path
from GoldEnv import GoldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):
    def _init():
        env = GoldEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init
    
if __name__ == '__main__':
    # === 설정 ===
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_finetune_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    log_dir = "./runs"
    
    # 불러올 Red 버전 모델 파일명 (확장자 .zip 제외 가능)
    pretrained_model_path = "./poke_26214400.zip" 

    print(f"=== Fine-tuning 시작! 세션 ID: {sess_id} ===")
    print(f"=== 기존 모델 로드 중: {pretrained_model_path} ===")

    env_config = { 
        'headless': True, 
        'save_final_state': True, 
        'early_stop': False, 
        'action_freq': 24, 
        'init_state': './init.state', 
        'max_steps': 2048 * 10, 
        'print_rewards': True, 
        'save_video': False, 
        'fast_video': True, 
        'session_path': sess_path,
        'gb_path': './PokeGold.gbc', # 골드 버전 롬 파일
        'debug': False, 
        'sim_frame_dist': 2_000_000.0, 
        'extra_buttons': False
    }

    num_cpu = 4 
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # === 콜백 설정 ===
    checkpoint_callback = CheckpointCallback(
        save_freq=500000 // num_cpu, 
        save_path=log_dir,
        name_prefix=f'poke_gold_finetune_{sess_id}'
    )
    tensorboard_callback = TensorboardCallback(log_dir=log_dir)
    callbacks = CallbackList([checkpoint_callback, tensorboard_callback])

    # === 모델 불러오기 (핵심 변경 부분) ===
    try:
        # custom_objects 설정을 통해 학습률(learning_rate) 등을 현재 설정으로 덮어씌웁니다.
        # 만약 에러가 난다면 env=env 부분을 빼고 로드한 뒤 model.set_env(env)를 해야할 수도 있습니다.
        model = PPO.load(
            pretrained_model_path, 
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            # 아래 파라미터들은 Red 학습 때와 달라도 덮어씌워집니다 (custom_objects 필요할 수 있음)
            custom_objects={
                "learning_rate": 0.0003, # Fine-tuning이라 학습률을 조금 낮추거나 그대로 유지
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.997,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
            }
        )
        print(">>> 모델 로드 성공! Red 버전의 지능을 이어받아 학습합니다.")
        
    except ValueError as e:
        print("\n[오류] 모델 구조가 일치하지 않습니다!")
        print(f"이유: {e}")
        print("Red 버전과 Gold 버전의 입력 데이터(Events 개수 등)가 달라서 호환되지 않을 확률이 높습니다.")
        print("이 경우, train.py를 사용하여 처음부터 학습(쌩 학습)해야 합니다.")
        env.close()
        exit()
    except Exception as e:
        print(f"\n[오류] 모델 로드 중 알 수 없는 에러 발생: {e}")
        env.close()
        exit()

    print("Fine-tuning 학습 시작...")
    
    try:
        model.learn(total_timesteps=100_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("\n중단됨. 저장 중...")
    
    model.save(f"{sess_path}/final_model_finetuned")
    env.close()