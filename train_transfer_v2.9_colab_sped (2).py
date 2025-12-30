import os
import uuid
import torch
import glob
import zipfile
import io
import mediapy as media
import numpy as np
from pathlib import Path
import time
from tensorboard import program
from GoldEnv import GoldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

from stable_baselines3.common.callbacks import BaseCallback

class GameStatsCallback(BaseCallback):
    """
    환경(GoldEnv)에서 보내준 info 데이터를 낚아채서
    텐서보드에 'game/...' 형태의 그래프로 그려주는 콜백
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.stats_buffer = []

    def _on_step(self) -> bool:
        # VecEnv(병렬 환경)에서는 infos가 리스트로 들어옵니다.
        # infos[0], infos[1]... 각 환경의 정보가 담겨 있습니다.
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "stats_badges" in info:
                # 여기서 원하는 지표를 텐서보드에 기록합니다.
                # 매 스텝 기록하면 그래프가 너무 지저분해지므로, 
                # 보통은 에피소드 단위나 일정 간격으로 기록하지만,
                # PPO는 배치가 크므로 매 스텝 로그를 남겨도 텐서보드가 알아서 스무딩해줍니다.
                
                self.logger.record("game/badges", info["stats_badges"])
                self.logger.record("game/level_sum", info["stats_level_sum"])
                self.logger.record("game/exploration", info["stats_explore"])
                self.logger.record("game/deaths", info["stats_deaths"])
                
                # 가끔 보고 싶은 정보 (현재 맵 ID 등은 그래프로 그리기 애매하므로 제외하거나 histogram 사용)
                
        return True

def make_env(rank, env_conf, seed=0):
    def _init():
        env = GoldEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

def find_latest_checkpoint(log_dir):
    list_of_files = glob.glob(os.path.join(log_dir, "*.zip"))
    if not list_of_files: return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env_config, save_freq, log_dir, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.sess_path = eval_env_config['session_path']
        self.record_config = eval_env_config.copy()
        self.record_config['headless'] = True
        self.record_config['save_video'] = True 
        self.record_config['instance_id'] = 'recorder'
        self.record_length =30000 
        self.record_config['max_steps'] = self.record_length + 1000 

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"\n🎥 [Video] 녹화 시작...")
            try:
                eval_env = GoldEnv(self.record_config)
                obs, _ = eval_env.reset()
                done = False
                truncated = False
                step_cnt = 0
                rollout_dir = os.path.join(self.sess_path, "rollouts")
                os.makedirs(rollout_dir, exist_ok=True)
                video_path = os.path.join(rollout_dir, f"video_step_{self.num_timesteps}.mp4")

                with media.VideoWriter(video_path, shape=(144, 160), fps=1200) as writer:
                    while not (done or truncated) and step_cnt < self.record_length:
                        action, _ = self.model.predict(obs, deterministic=False)
                        obs, _, done, truncated, _ = eval_env.step(action)
                        raw_screen = eval_env.pyboy.screen.ndarray[:, :, :3]
                        writer.add_image(raw_screen)
                        step_cnt += 1
                eval_env.close()
            except Exception as e:
                print(f"❌ [Video] 오류: {e}")
        return True

class UnfreezeCallback(BaseCallback):
    def __init__(self, unfreeze_step=500_000, verbose=0):
        super().__init__(verbose)
        self.unfreeze_step = unfreeze_step
        self.is_frozen = True

    def _on_step(self) -> bool:
        if self.is_frozen and self.num_timesteps >= self.unfreeze_step:
            print(f"\n🔥 [Unfreeze] {self.num_timesteps} 스텝 도달! Feature Extractor(CNN)를 녹입니다.")
            for param in self.model.policy.parameters():
                param.requires_grad = True
            self.is_frozen = False
        return True

def load_weights_from_zip(new_model, file_path, device="cpu"):
    print(f"\n=== 가중치 로드 시도: {file_path} ===")
    state_dict = None
    try:
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as archive:
                if 'policy.pth' in archive.namelist():
                    with archive.open('policy.pth') as f:
                        buffer = io.BytesIO(f.read())
                        state_dict = torch.load(buffer, map_location=device)
                else:
                    return PPO.load(file_path, env=new_model.get_env())
        else:
            state_dict = torch.load(file_path, map_location=device)

        if state_dict is not None:
            if not isinstance(state_dict, dict) and hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            new_model_dict = new_model.policy.state_dict()
            copied_count = 0
            for key in new_model_dict.keys():
                if key in state_dict and new_model_dict[key].shape == state_dict[key].shape:
                    new_model_dict[key] = state_dict[key]
                    copied_count += 1
            new_model.policy.load_state_dict(new_model_dict)
            print(f">>> 로드 완료: {copied_count}개 레이어 복사됨.")
    except Exception as e:
        print(f"[오류] 가중치 로드 실패: {e}")
    return new_model

def launch_tensorboard(log_dir):
    try:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006', '--bind_all'])
        url = tb.launch()
    except Exception:
        pass

if __name__ == '__main__':
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_gold_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    log_dir = "./runs"
    os.makedirs(log_dir, exist_ok=True)
    
    launch_tensorboard(log_dir)

    target_weights = find_latest_checkpoint(log_dir)
    is_resume = False
    base_red_path = "./poke_26214400/policy.pth" 

    if target_weights:
        print(f"🔄 [이어하기] 최신 체크포인트 발견: {target_weights}")
        is_resume = True
        unfreeze_step = 100_000 
    elif os.path.exists(base_red_path):
        print(f"🆕 [전이학습] Red 버전 가중치 로드")
        target_weights = base_red_path
        unfreeze_step = 500_000 
    else:
        print("🆕 [새로하기] 맨땅에 헤딩")
        target_weights = None
        unfreeze_step = 0

    env_config = {
        'headless': True, 
        'save_final_state': True, 
        'early_stop': False,
        'action_freq': 24, 
        'init_state': './init.state', 
        'max_steps': 4096 * 24,
        'print_rewards': True, 
        'save_video': False, 
        'fast_video': True,
        'session_path': sess_path, 
        'gb_path': './PokeGold.gbc',
        'debug': False, 
        'sim_frame_dist': 2_000_000.0, 
        'extra_buttons': False
    }

    num_cpu = min(os.cpu_count(), 8) 
    print(f"⚙️ 설정된 병렬 환경 개수: {num_cpu}")
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.00008, # [최적화] 학습률 4e-05
        n_steps=4096, 
        batch_size=128, 
        n_epochs=10, 
        gamma=0.997, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        ent_coef=0.01, 
    )

    if target_weights:
        model = load_weights_from_zip(model, target_weights)
        if not is_resume:
            for name, param in model.policy.named_parameters():
                if 'features_extractor' in name: param.requires_grad = False

    target_global_step = 100000
    save_freq = target_global_step // num_cpu
    print(f"💾 저장 및 녹화 주기: 전체 {target_global_step} 스텝 (CPU당 {save_freq} 스텝)")
    
    # ... (기존 콜백 리스트 정의 부분) ...
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix=f'gold_{sess_id}'),
        VideoRecorderCallback(env_config, save_freq=save_freq, log_dir=log_dir),
        
        # [추가] 우리가 만든 게임 스탯 로거
        GameStatsCallback(), 
        
        UnfreezeCallback(unfreeze_step=unfreeze_step)
    ])
    
    # model.learn(...)

    print(f"\n🚀 학습 시작! (Session: {sess_id})")
    model.learn(total_timesteps=50_000_000, callback=callbacks)
    model.save(f"{sess_path}/final_model")
    env.close()