import os
import uuid
import torch
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

# === [수정됨] device 기본값을 'cpu'로 변경 ===
def transfer_weights(new_model, old_weights_path, device="cpu"):
    """
    Red 버전의 .pth 파일에서 가중치를 직접 로드하여 Gold 모델에 이식합니다.
    """
    print(f"\n=== 가중치 이식 시작: {old_weights_path} ===")
    
    try:
        # map_location을 cpu로 명시
        old_state_dict = torch.load(old_weights_path, map_location=device)
        
        # 만약 로드한게 전체 모델 객체라면 state_dict만 추출
        if not isinstance(old_state_dict, dict):
             old_state_dict = old_state_dict.state_dict()
             
    except Exception as e:
        print(f"[치명적 오류] policy.pth 파일을 여는 데 실패했습니다: {e}")
        print(">>> 이식을 건너뛰고 쌩 학습을 진행합니다.")
        return new_model

    # 2. 새 모델(Gold)의 가중치 가져오기
    new_state_dict = new_model.policy.state_dict()
    
    # 3. 레이어 하나하나 비교하며 복사
    copied_count = 0
    skipped_count = 0
    
    print("\n[레이어 복사 상세]")
    for key in new_state_dict.keys():
        if key in old_state_dict:
            # 모양(Shape)이 정확히 일치하는지 확인
            if new_state_dict[key].shape == old_state_dict[key].shape:
                new_state_dict[key] = old_state_dict[key]
                copied_count += 1
            else:
                skipped_count += 1
                # print(f"스킵됨 (크기 불일치): {key}")
        else:
            skipped_count += 1
            # print(f"스킵됨 (Red에 없음): {key}")

    # 4. 합친 가중치를 새 모델에 적용
    new_model.policy.load_state_dict(new_state_dict)
    
    print(f"\n>>> 이식 성공! {copied_count}개의 레이어(시각 처리 등)를 복사했습니다.")
    print(f">>> {skipped_count}개의 레이어(맵/이벤트 판단)는 새로 학습합니다.")
    
    return new_model

if __name__ == '__main__': 
    # === 설정 ===
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_transfer_{sess_id}')
    sess_path.mkdir(exist_ok=True) 
    log_dir = "./runs" 
    
    # [중요] 본인의 경로에 맞게 수정하세요! 
    # (아까 에러 메시지를 보니 ./poke_26214400/policy.pth 경로가 맞는 것 같습니다) 
    red_weights_path = "./poke_26214400/policy.pth"  

    print(f"=== 전이 학습(Transfer Learning) 시작! 세션 ID: {sess_id} ===")

    env_config = { 
        'headless': False,  
        'save_final_state': True,  
        'early_stop': False,  
        'action_freq': 24,   
        'init_state': './init.state',   
        'max_steps': 2048 * 10,   
        'print_rewards': True,    
        'save_video': False,    
        'fast_video': True,    
        'session_path': sess_path,   
        'gb_path': './PokeGold.gbc',    
        'debug': False,   
        'sim_frame_dist': 2_000_000.0,   
        'extra_buttons': False    
    }

    num_cpu = 4
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir, 
        learning_rate=0.0003, 
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.997,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # === 파일 존재 확인 후 이식 ===
    if os.path.exists(red_weights_path):
        model = transfer_weights(model, red_weights_path, device="cpu")
    else:
        print(f"!!! 오류: {red_weights_path} 파일을 찾을 수 없습니다.")
        print("!!! 경로를 확인해주세요.")
        env.close()
        exit()

    checkpoint_callback = CheckpointCallback(
        save_freq=500000 // num_cpu, 
        save_path=log_dir,
        name_prefix=f'poke_gold_transfer_{sess_id}'
    )
    tensorboard_callback = TensorboardCallback(log_dir=log_dir)
    callbacks = CallbackList([checkpoint_callback, tensorboard_callback])

    print("학습 시작...")
    try:
        model.learn(total_timesteps=100_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("\n중단됨. 저장 중...")
    
    model.save(f"{sess_path}/final_model_transfer")
    env.close()