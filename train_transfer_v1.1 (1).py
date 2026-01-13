import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
import uuid
import glob
import zipfile
import io
import torch
import cv2
import numpy as np
from collections import defaultdict
from tensorboard import program

from GoldEnv import GoldEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

# 윈도우 콘솔 종료 방지
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

# ==========================================
# 1. 영상 녹화 콜백
# ==========================================
class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env_config, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.sess_path = eval_env_config['session_path']
        
        self.record_config = eval_env_config.copy()
        self.record_config['headless'] = True
        self.record_config['save_video'] = True
        self.record_config['instance_id'] = 'recorder'
        
        self.record_length = 7000 
        self.record_config['max_steps'] = self.record_length + 500

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"\n🎥 [Video] 녹화 시작... (Step: {self.num_timesteps})")
            try:
                eval_env = GoldEnv(self.record_config)
                obs, _ = eval_env.reset()
                done = False
                truncated = False
                step_cnt = 0
                
                rollout_dir = os.path.join(self.sess_path, "rollouts")
                os.makedirs(rollout_dir, exist_ok=True)
                video_path = os.path.join(rollout_dir, f"video_step_{self.num_timesteps}.mp4")

                dummy_screen = eval_env.pyboy.screen.ndarray[:, :, :3]
                height, width, channels = dummy_screen.shape
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))

                if not writer.isOpened():
                    print("❌ [Video] VideoWriter를 열 수 없습니다.")
                    eval_env.close()
                    return True

                while not (done or truncated) and step_cnt < self.record_length:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, done, truncated, _ = eval_env.step(action)
                    
                    raw_screen = eval_env.pyboy.screen.ndarray[:, :, :3]
                    frame_bgr = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
                    
                    writer.write(frame_bgr)
                    step_cnt += 1
                    
                    if step_cnt % 1000 == 0:
                         print(f"   -> {step_cnt} / {self.record_length} 프레임 저장 중...", end='\r')

                writer.release()
                eval_env.close()
                print(f"\n✅ [Video] 저장 완료: {video_path}")
            
            except Exception as e:
                print(f"\n❌ [Video] 오류 발생: {e}")
                
        return True

# ==========================================
# 2. 통계 콜백 (여기서 로그 키를 관리합니다)
# ==========================================
class GameStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        
        # ✅ reward_new_map 포함 여부 확인
        self.keys = [
            "reward_total", "reward_explore", "reward_level", "reward_badge",
            "reward_event", "reward_heal", "reward_exp", "reward_dmg", 
            "reward_stuck", "reward_battle", "reward_new_map", 
            "stats_explore", "stats_level_sum", "stats_steps"
        ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos: return True
        for key in self.keys:
            vals = [info[key] for info in infos if key in info]
            if vals:
                self.sums[key] += float(np.sum(vals))
                self.counts[key] += int(len(vals))
        return True

    def _on_rollout_end(self) -> None:
        for key in self.keys:
            c = self.counts.get(key, 0)
            if c > 0:
                self.logger.record(f"game/{key}", self.sums[key] / c)
        self.sums.clear()
        self.counts.clear()

class UnfreezeCallback(BaseCallback):
    def __init__(self, unfreeze_step=100_000, verbose=0):
        super().__init__(verbose)
        self.unfreeze_step = unfreeze_step
        self.done = False

    def _on_step(self) -> bool:
        if (not self.done) and self.num_timesteps >= self.unfreeze_step:
            print(f"\n🔥 [Unfreeze] {self.num_timesteps} steps! CNN 가중치 잠금 해제.")
            for param in self.model.policy.features_extractor.parameters():
                param.requires_grad = True
            self.done = True
        return True

# ==========================================
# 3. 유틸리티 함수
# ==========================================
def make_env(rank, env_conf, seed=0):
    def _init():
        set_random_seed(seed + rank)
        conf = env_conf.copy()
        conf["instance_id"] = f"{rank}_{str(uuid.uuid4())[:4]}"
        conf["rank"] = rank
        env = GoldEnv(conf)
        env.reset(seed=seed + rank)
        return env
    return _init

def safe_load_weights(model, file_path, device="cpu"):
    print(f"\n🔧 가중치 이식 시도: {file_path}")
    if not os.path.exists(file_path):
        print("⚠️ 파일 없음. 스킵합니다.")
        return model

    try:
        params = None
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as archive:
                if "policy.pth" in archive.namelist():
                    with archive.open("policy.pth") as f:
                        params = torch.load(io.BytesIO(f.read()), map_location="cpu")
        else:
            params = torch.load(file_path, map_location="cpu")

        if params is None:
            temp_model = PPO.load(file_path, device="cpu")
            params = temp_model.policy.state_dict()

        if hasattr(params, "state_dict"):
            params = params.state_dict()

        current_state = model.policy.state_dict()
        copied = 0
        skipped = 0

        for name, param in params.items():
            if name in current_state:
                if current_state[name].shape == param.shape:
                    current_state[name].copy_(param)
                    copied += 1
                else:
                    skipped += 1

        model.policy.load_state_dict(current_state)
        print(f"✅ 가중치 로드 완료! (성공: {copied} / 스킵: {skipped})")

    except Exception as e:
        print(f"❌ 가중치 로드 실패: {e}")

    return model

# ==========================================
# 4. 메인 실행
# ==========================================
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    sess_id = str(uuid.uuid4())[:8]
    tb_name = f"PPO_{sess_id}"

    # 로그 폴더
    log_dir = "C:/pokey_logs" 
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(current_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"ℹ️ 로그 저장 경로: {log_dir}")
    
    try:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006'])
        url = tb.launch()
        print(f"📊 TensorBoard 시작: {url}")
    except:
        print("⚠️ 텐서보드 자동 실행 실패 (무시 가능)")

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    target_weights = None
    
    if checkpoints:
        target_weights = max(checkpoints, key=os.path.getmtime)
        print(f"🔄 가장 최근 체크포인트 선택됨: {os.path.basename(target_weights)}")

    red_weights = os.path.join(current_dir, "poke_red_weights.pth")

    rom_path = os.path.join(current_dir, 'PokeGold.gbc')
    init_state_path = os.path.join(current_dir, 'init.state')

    if not os.path.exists(rom_path):
        print(f"❌ [Error] 롬 파일이 없습니다: {rom_path}")
        exit()
    if not os.path.exists(init_state_path):
        print(f"❌ [Error] 초기 상태 파일이 없습니다: {init_state_path}")
        exit()

    env_config = {
        'headless': True,
        'save_final_state': True,
        'action_freq': 24,
        'init_state': init_state_path,
        'max_steps': 4096 * 15, 
        'save_video': False, 
        'fast_video': False,
        'print_rewards': False,
        'session_path': os.path.join(current_dir, f'session_{sess_id}'),
        'gb_path': rom_path,
        'explore_weight': 2.1,
        'reward_scale': 1.0,
    }

    num_cpu = min(os.cpu_count(), 8) 
    print(f"⚙️ 병렬 환경 개수: {num_cpu}")
    
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = VecMonitor(env, filename=os.path.join(log_dir, f"monitor_{sess_id}.csv"))

    model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir, 
            learning_rate=0.00008,
            n_steps=2048,
            batch_size=128,
            n_epochs=6,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.015,
            target_kl=0.02,
        )

    if target_weights:
        print(f"♻️ 이어하기 모드: {target_weights}")
        model = PPO.load(
            target_weights,
            env=env,
            tensorboard_log=log_dir,
            device="auto",
            verbose=1,
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir, 
            learning_rate=0.00008,
            n_steps=4096,
            batch_size=128,
            n_epochs=6,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.015,
        )
        if os.path.exists(red_weights):
            print("🚀 전이 학습 모드 (Red -> Gold)")
            model = safe_load_weights(model, red_weights)


    print(f"\n🎮 학습 시작! Session: {sess_id}")

    save_freq = 50000 // num_cpu

    callbacks = CallbackList([
        CheckpointCallback(save_freq=save_freq, save_path=checkpoint_dir, name_prefix="gold_auto"),
        #VideoRecorderCallback(env_config, save_freq=save_freq),
        GameStatsCallback(),
    ])

    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=callbacks,
            tb_log_name=tb_name,
            log_interval=1,
        )
    except KeyboardInterrupt:
        print("\n🛑 학습 중단 요청. 저장 중...")

    final_path = os.path.join(checkpoint_dir, f"final_model_{sess_id}")
    model.save(final_path)
    print("✅ 저장 완료 및 종료.")
    env.close()
