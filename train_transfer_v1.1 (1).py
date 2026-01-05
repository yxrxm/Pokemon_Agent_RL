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
from stable_baselines3.common.callbacks import CheckpointCallback  # <--- 이 줄 추가!

class GameStatsCallback(BaseCallback):
    """
    환경(GoldEnv)에서 보내준 info 데이터를 낚아채서
    텐서보드에 'game/...' 형태의 그래프로 그려주는 콜백
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        for info in infos:
            # 1. 기본 통계 (GoldEnv의 info 키와 일치해야 함)
            if "stats_badges" in info:
                self.logger.record("game/badges", info["stats_badges"])
            if "stats_level_sum" in info:
                total_level = info["stats_level_sum"] + 5
                self.logger.record("game/level_sum", total_level)
            if "stats_explore" in info:
                self.logger.record("game/exploration", info["stats_explore"])
            if "stats_deaths" in info:
                self.logger.record("game/deaths", info["stats_deaths"])
            
            # [수정됨] GoldEnv에서 'reward_total'로 보냄
            if "reward_total" in info:
                self.logger.record("game/total_reward", info["reward_total"])
                
            # 2. 세부 보상 항목 기록
            # GoldEnv가 'reward_exp', 'reward_explore' 등의 키로 보냅니다.
            for key, value in info.items():
                if key.startswith("reward_") and key != "reward_total":
                    # 예: reward_exp -> game/reward_exp
                    self.logger.record(f"game/{key}", value)
                
        return True

def make_env(rank, env_conf, seed=0):
    def _init():
        # [수정] env_conf가 딕셔너리이므로 복사해서 사용 (안전성)
        conf = env_conf.copy()
        # 멀티프로세싱 시 각 환경마다 별도의 instance_id 부여
        conf["instance_id"] = f"{rank}_{str(uuid.uuid4())[:4]}"
        env = GoldEnv(conf)
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
        self.record_config['headless'] = True # 녹화용은 무조건 Headless
        self.record_config['save_video'] = False # GoldEnv 자체 녹화 기능 대신 여기서 직접 제어
        self.record_config['instance_id'] = 'recorder'
        self.record_length = 10000 # 녹화 길이 조절
        self.record_config['max_steps'] = self.record_length + 100

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"\n🎥 [Video] 녹화 시작 (Step: {self.num_timesteps})...")
            try:
                # 녹화용 환경 생성
                eval_env = GoldEnv(self.record_config)
                obs, _ = eval_env.reset()
                done = False
                truncated = False
                step_cnt = 0
                
                rollout_dir = os.path.join(self.sess_path, "rollouts")
                os.makedirs(rollout_dir, exist_ok=True)
                video_path = os.path.join(rollout_dir, f"video_step_{self.num_timesteps}.mp4")

                # Mediapy로 녹화
                with media.VideoWriter(video_path, shape=(144, 160), fps=60) as writer:
                    while not (done or truncated) and step_cnt < self.record_length:
                        action, _ = self.model.predict(obs, deterministic=False)
                        obs, _, done, truncated, _ = eval_env.step(action)
                        
                        # PyBoy 화면 가져오기
                        raw_screen = eval_env.render(reduce_res=False)[:, :, 0]
                        writer.add_image(raw_screen)
                        step_cnt += 1
                
                eval_env.close()
                print(f"✅ [Video] 저장 완료: {video_path}")
            except Exception as e:
                print(f"❌ [Video] 오류 발생: {e}")
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
                # [중요] Shape이 맞을 때만 복사 (Level Input 크기가 바뀌었으므로 필수 체크)
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
        print(f"📊 TensorBoard 실행됨: {url}")
    except Exception:
        print("⚠️ TensorBoard 실행 실패 (이미 실행 중이거나 설치되지 않음)")

# [수정 위치] if __name__ == '__main__': 바로 아래 부분

if __name__ == '__main__':
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_gold_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    log_dir = "./runs"
    os.makedirs(log_dir, exist_ok=True)
    
    launch_tensorboard(log_dir)

    # 1. 구글 드라이브 경로 설정
    drive_checkpoint_dir = "/content/drive/MyDrive/PokeyProjectGoldVer_hwlee_20251124/checkpoints"
    os.makedirs(drive_checkpoint_dir, exist_ok=True)

    # 2. [핵심 수정] 로컬과 드라이브 양쪽을 다 뒤져서 '가장 최신' 하나를 뽑음
    print("🔍 체크포인트 검색 중... (로컬 + 구글드라이브)")
    
    local_files = glob.glob(os.path.join(log_dir, "*.zip"))
    drive_files = glob.glob(os.path.join(drive_checkpoint_dir, "*.zip"))
    
    # 두 리스트 합치기
    all_files = local_files + drive_files
    
    target_weights = None
    if all_files:
        # 생성 시간(getctime) 기준으로 정렬해서 제일 마지막 거(가장 최신) 선택
        # 리눅스 환경에선 getmtime(수정시간)이 더 안전할 수 있어 getmtime 사용
        latest_file = max(all_files, key=os.path.getmtime)
        target_weights = latest_file
        print(f"👉 발견된 가장 최신 파일: {target_weights}")
    else:
        print("👉 발견된 체크포인트 없음. 새로 시작합니다.")

    # ... (이후 로직은 동일) ...
    is_resume = False
    base_red_path = "./poke_26214400/policy.pth" 

    if target_weights:
        print(f"🔄 [이어하기] 로드: {target_weights}")
        is_resume = True
        unfreeze_step = 100_000 
    # ... (아래는 기존 코드 그대로 유지)
    elif os.path.exists(base_red_path):
        print(f"🆕 [전이학습] Red 버전 가중치 로드")
        target_weights = base_red_path
        unfreeze_step = 500_000 
    else:
        print("🆕 [새로하기] 맨땅에 헤딩")
        target_weights = None
        unfreeze_step = 0

    base_path = "/content/drive/MyDrive/PokeyProjectGoldVer_hwlee_20251124"
    
    env_config = {
        'headless': True, 
        'save_final_state': True, 
        'action_freq': 24, 
        'init_state': os.path.join(base_path, 'init.state'), 
        'max_steps': 4096 * 8, 
        'print_rewards': False, 
        'save_video': False, 
        'fast_video': False,
        'session_path': str(sess_path), 
        'gb_path': os.path.join(base_path, 'PokeGold.gbc'), # 절대 경로 추천
        'explore_weight': 2.0, 
        'reward_scale': 1.0,
        'event_flags_start': 0xD7B7 # [중요] 이벤트 주소 수정 확인!
    }

    num_cpu = min(os.cpu_count(), 8)
    print(f"⚙️ 설정된 병렬 환경 개수: {num_cpu}")
    
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    model = PPO(
        "MultiInputPolicy",
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
    )

    if target_weights:
        model = load_weights_from_zip(model, target_weights)
        if not is_resume:
            print("❄️ Feature Extractor(CNN)를 얼립니다.")
            for name, param in model.policy.named_parameters():
                if 'features_extractor' in name: param.requires_grad = False

    target_global_step = 100_000 
    save_freq = max(target_global_step // num_cpu, 1)
    
    print(f"💾 저장 및 녹화 주기: 전체 {target_global_step} 스텝 (Env당 {save_freq} 스텝)")
    
    # [핵심 수정] 모든 콜백을 하나의 리스트로 통합!
    # 여기에 CheckpointCallback을 포함시켜야 합니다.
    callbacks = CallbackList([
        # 1. 구글 드라이브 자동 저장 (50,000 스텝마다)
        CheckpointCallback(
            save_freq=50000 // num_cpu,  # 병렬 환경 고려 (전체 약 5만 스텝)
            save_path=drive_checkpoint_dir, 
            name_prefix="gold_auto"
        ),
        # 2. 비디오 녹화
        VideoRecorderCallback(env_config, save_freq=save_freq, log_dir=log_dir),
        # 3. 게임 통계 로그
        GameStatsCallback(), 
        # 4. CNN 동결 해제
        UnfreezeCallback(unfreeze_step=unfreeze_step)
    ])
    
    print(f"\n🚀 학습 시작! (Session: {sess_id})")
    try:
        model.learn(
            total_timesteps=env_config['max_steps'] * 1000, # [수정] CONF -> env_config
            callback=callbacks  # [수정] 통합된 리스트(callbacks)를 전달
        )
    except KeyboardInterrupt:
        print("\n🛑 학습 중단 요청됨. 모델 저장 중...")
    
    # 마지막 저장 (혹시 모르니 드라이브에도 저장)
    final_save_path = os.path.join(drive_checkpoint_dir, f"final_model_{sess_id}")
    model.save(final_save_path)
    print(f"✅ 최종 모델 저장 완료: {final_save_path}")
    
    env.close()
    print("✅ 학습 종료 및 환경 닫힘.")