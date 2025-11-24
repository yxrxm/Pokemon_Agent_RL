import os
import uuid
import torch
import torch.nn as nn
from pathlib import Path
from GoldEnv import GoldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):
    def _init():
        env = GoldEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

# === [1] 일반화 구현: 가져온 지식 얼리기 (Freeze) ===
def freeze_layers(model, freeze_modules=['features_extractor']):
    print(f"\n🥶 [일반화] '{freeze_modules}' 관련 신경망을 얼립니다 (업데이트 차단).")
    print("   -> Red 버전에서 배운 '보는 능력'을 그대로 유지합니다.")
    
    for name, param in model.policy.named_parameters():
        # 지정한 모듈 이름이 포함된 레이어라면
        if any(module in name for module in freeze_modules):
            param.requires_grad = False # 학습되지 않도록 잠금
            # print(f"   - Locked: {name}") # 너무 길어서 주석

# === [2] 변별 후 융합: 나중에 녹이기 (Unfreeze Callback) ===
class UnfreezeCallback(BaseCallback):
    def __init__(self, unfreeze_step=2_000_000, verbose=0):
        super().__init__(verbose)
        self.unfreeze_step = unfreeze_step
        self.is_frozen = True

    def _on_step(self) -> bool:
        # 지정된 스텝이 지나면 얼음 땡!
        if self.is_frozen and self.num_timesteps >= self.unfreeze_step:
            print(f"\n🔥 [변별 완료] {self.num_timesteps} 스텝 도달! 신경망을 녹입니다.")
            print("   -> 이제 Gold 환경의 미세한 차이(색감 등)까지 학습하기 시작합니다.")
            
            # 모든 파라미터 잠금 해제
            for param in self.model.policy.parameters():
                param.requires_grad = True
            
            # 학습률을 조금 낮춰서 섬세하게 튜닝하도록 변경 (선택 사항)
            self.model.learning_rate = 0.0001 
            self.is_frozen = False
        return True

# 기존 가중치 이식 함수 (device='cpu' 수정본)
def transfer_weights(new_model, old_weights_path, device="cpu"):
    print(f"\n=== 가중치 이식 시작: {old_weights_path} ===")
    try:
        old_state_dict = torch.load(old_weights_path, map_location=device)
        if not isinstance(old_state_dict, dict):
             old_state_dict = old_state_dict.state_dict()
    except Exception as e:
        print(f"[오류] 파일 로드 실패: {e}")
        return new_model

    new_state_dict = new_model.policy.state_dict()
    copied_count = 0
    
    # 모양이 같은 것만 복사 (눈, 운동신경 등)
    for key in new_state_dict.keys():
        if key in old_state_dict and new_state_dict[key].shape == old_state_dict[key].shape:
            new_state_dict[key] = old_state_dict[key]
            copied_count += 1

    new_model.policy.load_state_dict(new_state_dict)
    print(f">>> 이식 완료: {copied_count}개 레이어 복사됨.")
    return new_model

if __name__ == '__main__':
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_smart_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    log_dir = "./runs"
    
    red_weights_path = "./poke_26214400/policy.pth"

    print(f"=== 스마트 전이 학습 시작! 세션 ID: {sess_id} ===")

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

    # 1. 가중치 이식
    if os.path.exists(red_weights_path):
        model = transfer_weights(model, red_weights_path, device="cpu")
        
        # 2. [핵심] 가져온 지식(CNN) 얼리기! (일반화 적용)
        # 'features_extractor'는 CNN(시각 처리) 부분을 의미합니다.
        freeze_layers(model, freeze_modules=['features_extractor'])
        
    else:
        print("!!! Red 모델을 찾을 수 없어 쌩 학습합니다.")

    # 콜백 설정
    checkpoint_callback = CheckpointCallback(save_freq=500000 // num_cpu, save_path=log_dir, name_prefix=f'poke_smart_{sess_id}')
    tensorboard_callback = TensorboardCallback(log_dir=log_dir)
    
    # 3. [핵심] 200만 스텝 후에 녹이는 콜백 추가
    unfreeze_callback = UnfreezeCallback(unfreeze_step=2_000_000) 
    
    callbacks = CallbackList([checkpoint_callback, tensorboard_callback, unfreeze_callback])

    print("학습 시작...")
    try:
        model.learn(total_timesteps=100_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("\n중단됨.")
    
    model.save(f"{sess_path}/final_model_smart")
    env.close()