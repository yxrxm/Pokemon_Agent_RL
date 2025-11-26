# 동영상 저장 = 체크포인트

import os
import uuid
import torch
import glob
import zipfile
import io
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

# === [기능 추가] 가장 최신 체크포인트 찾기 ===
def find_latest_checkpoint(log_dir):
    list_of_files = glob.glob(os.path.join(log_dir, "*.zip"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# === [기능 추가] 체크포인트 시점에 맞춰 영상을 녹화하는 콜백 ===
class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env_config, save_freq, log_dir, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_dir = log_dir
        
        # 영상 녹화용 설정 강제 적용
        # max_steps를 짧게(예: 2048) 설정하여 영상이 너무 길어지지 않게 합니다.
        record_config = eval_env_config.copy()
        record_config['save_video'] = True
        record_config['headless'] = True # 영상만 남기면 되므로 화면은 안 띄움
        record_config['max_steps'] = 2048 # 영상 길이 (약 2000프레임)
        record_config['instance_id'] = 'recorder' # 파일명 식별용
        
        # 녹화용 별도 환경 생성 (학습 환경과 분리)
        self.eval_env = GoldEnv(record_config)

    def _on_step(self) -> bool:
        # save_freq 마다 실행
        if self.n_calls % self.save_freq == 0:
            print(f"\n🎥 [Video] {self.num_timesteps} 스텝: 체크포인트 영상 녹화를 시작합니다...")
            
            # 녹화용 환경 리셋 (이때 내부적으로 비디오 파일이 열림)
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            
            # 정해진 max_steps 만큼 플레이 (모델이 행동 결정)
            while not (done or truncated):
                # 현재 학습된 모델로 행동 예측
                action, _ = self.model.predict(obs, deterministic=False)
                obs, _, done, truncated, _ = self.eval_env.step(action)
            
            # 에피소드가 끝나면 GoldEnv 내부에서 writer.close()가 호출되어 영상이 저장됨
            print(f"✅ [Video] 영상 저장 완료.")
            
        return True

# === [1] Freeze ===
def freeze_layers(model, freeze_modules=['features_extractor']):
    print(f"\n🥶 [Freeze] '{freeze_modules}' 관련 신경망을 얼립니다.")
    for name, param in model.policy.named_parameters():
        if any(module in name for module in freeze_modules):
            param.requires_grad = False

# === [2] Unfreeze Callback ===
class UnfreezeCallback(BaseCallback):
    def __init__(self, unfreeze_step=2_000_000, verbose=0):
        super().__init__(verbose)
        self.unfreeze_step = unfreeze_step
        self.is_frozen = True

    def _on_step(self) -> bool:
        if self.is_frozen and self.num_timesteps >= self.unfreeze_step:
            print(f"\n🔥 [Unfreeze] {self.num_timesteps} 스텝 도달! 신경망을 녹입니다.")
            for param in self.model.policy.parameters():
                param.requires_grad = True
            self.model.learning_rate = 0.0001 
            self.is_frozen = False
        return True

# === Zip 파일 내부 가중치 로드 함수 ===
def load_weights_from_zip(new_model, file_path, device="cpu"):
    print(f"\n=== 가중치 로드 시도: {file_path} ===")
    state_dict = None
    try:
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as archive:
                if 'policy.pth' in archive.namelist():
                    print("   -> Zip 내부의 'policy.pth'를 발견했습니다.")
                    with archive.open('policy.pth') as f:
                        buffer = io.BytesIO(f.read())
                        state_dict = torch.load(buffer, map_location=device)
                else:
                    return new_model
        else:
            state_dict = torch.load(file_path, map_location=device)

        if not isinstance(state_dict, dict):
             if hasattr(state_dict, 'state_dict'):
                 state_dict = state_dict.state_dict()
             else:
                 return new_model

    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        return new_model

    new_model_dict = new_model.policy.state_dict()
    copied_count = 0
    for key in new_model_dict.keys():
        if key in state_dict and new_model_dict[key].shape == state_dict[key].shape:
            new_model_dict[key] = state_dict[key]
            copied_count += 1

    new_model.policy.load_state_dict(new_model_dict)
    print(f">>> 로드 완료: {copied_count}개 레이어 복사됨.")
    return new_model

if __name__ == '__main__':
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_smart_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    log_dir = "./runs"
    os.makedirs(log_dir, exist_ok=True)
    
    base_red_path = "./poke_26214400/policy.pth" 

    target_weights = find_latest_checkpoint(log_dir)
    is_resume = True
    
    if target_weights is None:
        target_weights = base_red_path
        is_resume = False
        print("   -> 저장된 Gold 체크포인트가 없습니다. Red 모델을 찾습니다.")
    else:
        print(f"   -> 최신 체크포인트 발견! 이어하기 모드: {target_weights}")

    print(f"=== 학습 시작! 세션 ID: {sess_id} ===")

    # 학습용 환경 설정 (영상 저장 끔)
    env_config = {
        'headless': True, 
        'save_final_state': True, 
        'early_stop': False,
        'action_freq': 24, 
        'init_state': './init.state', 
        'max_steps': 2048 * 10, 
        'print_rewards': True, 
        'save_video': False,  # 학습 중에는 영상 저장 안 함 (속도 저하 방지)
        'fast_video': True, 
        'session_path': sess_path,
        'gb_path': './PokeGold.gbc', 
        'debug': False, 
        'sim_frame_dist': 2_000_000.0, 
        'extra_buttons': False
    }

    # 코랩 CPU 개수에 맞춰서 자동으로 설정하거나, 안전하게 2~4로 고정
    logical_cpu_count = os.cpu_count()
    # 너무 많이 잡으면 오버헤드로 느려지므로, 최대 4~8개로 제한
    # T4 무료 버전은 보통 2개이므로 2~4 사이가 적당함
    num_cpu = min(logical_cpu_count, 8) 
    print(f"설정된 병렬 환경 개수: {num_cpu}")
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

    if target_weights and os.path.exists(target_weights):
        model = load_weights_from_zip(model, target_weights, device="cpu")
        freeze_layers(model, freeze_modules=['features_extractor'])
    else:
        print("!!! 로드할 모델 파일이 없습니다. 쌩 학습 시작.")

    # 저장 빈도 설정 (총 10만 스텝마다)
    save_freq_steps = 100000 // num_cpu

    # 1. 체크포인트 콜백 (.zip 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_steps, 
        save_path=log_dir, 
        name_prefix=f'poke_smart_{sess_id}'
    )
    
    # 2. 비디오 녹화 콜백 (.mp4 저장) - 새로 추가됨
    video_callback = VideoRecorderCallback(
        eval_env_config=env_config,
        save_freq=save_freq_steps,
        log_dir=log_dir
    )

    # 3. 텐서보드 콜백
    tensorboard_callback = TensorboardCallback(log_dir=log_dir)
    
    # 4. 언프리즈 콜백
    unfreeze_callback = UnfreezeCallback(unfreeze_step=500_000)
    
    # 콜백 리스트 합체
    callbacks = CallbackList([
        checkpoint_callback, 
        video_callback,  # <--- 영상 녹화 추가
        tensorboard_callback, 
        unfreeze_callback
    ])

    print("학습 프로세스 시작...")
    try:
        model.learn(total_timesteps=100_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("\n중단됨.")
    
    model.save(f"{sess_path}/final_model_smart")
    env.close()