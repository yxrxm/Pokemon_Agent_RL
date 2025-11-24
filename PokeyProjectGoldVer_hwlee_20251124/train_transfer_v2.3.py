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
    # runs 폴더 내의 모든 zip 파일 검색
    list_of_files = glob.glob(os.path.join(log_dir, "*.zip"))
    if not list_of_files:
        return None
    # 생성 시간(getctime) 또는 수정 시간(getmtime) 기준으로 가장 최신 파일 반환
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# === [1] 일반화 구현: 가져온 지식 얼리기 (Freeze) ===
def freeze_layers(model, freeze_modules=['features_extractor']):
    print(f"\n🥶 [Freeze] '{freeze_modules}' 관련 신경망을 얼립니다.")
    for name, param in model.policy.named_parameters():
        if any(module in name for module in freeze_modules):
            param.requires_grad = False

# === [2] 변별 후 융합: 나중에 녹이기 (Unfreeze Callback) ===
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

# === [수정됨] Zip 파일 내부의 pth를 직접 로드하는 함수 ===
def load_weights_from_zip(new_model, file_path, device="cpu"):
    print(f"\n=== 가중치 로드 시도: {file_path} ===")
    
    state_dict = None
    
    try:
        # 1. ZIP 파일인 경우
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as archive:
                # SB3 저장 방식은 내부에 'policy.pth'가 있습니다.
                if 'policy.pth' in archive.namelist():
                    print("   -> Zip 내부의 'policy.pth'를 발견했습니다.")
                    with archive.open('policy.pth') as f:
                        # 메모리 버퍼로 읽어서 torch.load에 전달 (압축 해제 불필요)
                        buffer = io.BytesIO(f.read())
                        state_dict = torch.load(buffer, map_location=device)
                else:
                    print("   -> Zip 내부에 policy.pth가 없습니다. (전체 모델 파일일 수 있음)")
                    # 전체 모델인 경우 파라미터만 추출 시도 (복잡하므로 여기선 패스하거나 추가 로직 필요)
                    return new_model
        
        # 2. 그냥 PTH 파일인 경우
        else:
            state_dict = torch.load(file_path, map_location=device)

        # 딕셔너리 형태 확인
        if not isinstance(state_dict, dict):
             if hasattr(state_dict, 'state_dict'):
                 state_dict = state_dict.state_dict()
             else:
                 print("   -> 로드된 객체가 딕셔너리가 아닙니다.")
                 return new_model

    except Exception as e:
        print(f"[오류] 파일 읽기 실패: {e}")
        return new_model

    # 가중치 이식 (Shape 맞는 것만)
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
    os.makedirs(log_dir, exist_ok=True) # 폴더가 없으면 생성
    
    # [중요] 여기에 Red 버전 원본 파일 경로를 적으세요! (폴더 말고 파일)
    # 만약 runs 폴더에 아무것도 없으면 이 파일을 씁니다.
    base_red_path = "./poke_26214400/policy.pth" 

    # === [핵심] 자동 로드 로직 ===
    # 1순위: runs 폴더의 최신 Gold 체크포인트 (이어하기)
    target_weights = find_latest_checkpoint(log_dir)
    is_resume = True
    
    # 2순위: 없으면 Red 모델 (전이 학습)
    if target_weights is None:
        target_weights = base_red_path
        is_resume = False
        print("   -> 저장된 Gold 체크포인트가 없습니다. Red 모델을 찾습니다.")
    else:
        print(f"   -> 최신 체크포인트 발견! 이어하기 모드: {target_weights}")

    # ----------------------------------

    print(f"=== 학습 시작! 세션 ID: {sess_id} ===")

    env_config = {
        'headless': False,  
        'save_final_state': True, 
        'early_stop': False,
        'action_freq': 24, 
        'init_state': './init.state', 
        'max_steps': 2048 * 1000, 
        'print_rewards': True, 
        'save_video': False, 
        'fast_video': True, 
        'session_path': sess_path,
        'gb_path': './PokeGold.gbc', 
        'debug': False, 
        'sim_frame_dist': 2_000_000.0, 
        'extra_buttons': False
    }

    num_cpu = 2
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

    # === 가중치 로드 및 Freeze 적용 ===
    if target_weights and os.path.exists(target_weights):
        model = load_weights_from_zip(model, target_weights, device="cpu")
        
        # 이어하기든, 전이학습이든 일단 200만 스텝까지는 Freeze로 안정화 추천
        # (이미 많이 학습된 모델이라면 금방 Unfreeze 될 것임)
        freeze_layers(model, freeze_modules=['features_extractor'])
    else:
        print("!!! 로드할 모델 파일이 없습니다. 쌩 학습 시작.")

    # 콜백 설정
    checkpoint_callback = CheckpointCallback(save_freq=500000 // num_cpu, save_path=log_dir, name_prefix=f'poke_smart_{sess_id}')
    tensorboard_callback = TensorboardCallback(log_dir=log_dir)
    unfreeze_callback = UnfreezeCallback(unfreeze_step=2_000_000) 
    
    callbacks = CallbackList([checkpoint_callback, tensorboard_callback, unfreeze_callback])

    print("학습 프로세스 시작...")
    try:
        model.learn(total_timesteps=100_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("\n중단됨.")
    
    model.save(f"{sess_path}/final_model_smart")
    env.close()