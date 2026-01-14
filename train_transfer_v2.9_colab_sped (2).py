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
import cv2

# 덮어써서 병렬로 하면 마지막거만 보임
# class GameStatsCallback(BaseCallback):
#     """
#     환경(GoldEnv)에서 보내준 info 데이터를 낚아채서
#     텐서보드에 'game/...' 형태의 그래프로 그려주는 콜백
#     """
#     def __init__(self, verbose=0):
#         super().__init__(verbose)

#     def _on_step(self) -> bool:
#         infos = self.locals.get("infos", [])
        
#         for info in infos:
#             if "reward_total" in info:
#                 # 1. 기존 게임 스탯
#                 self.logger.record("game/badges", info["stats_badges"])
#                 self.logger.record("game/level_sum", info["stats_level_sum"])
#                 self.logger.record("game/exploration", info["stats_explore"])
#                 self.logger.record("game/deaths", info["stats_deaths"])
                
#                 # 2. [추가] 보상(Reward) 상세 내역
#                 # 이걸 보면 "아, 방금 그래프가 튄 건 탐험 때문이구나" 하고 알 수 있습니다.
#                 self.logger.record("reward/total", info["reward_total"])
#                 self.logger.record("reward/exploration", info["reward_explore"])
#                 self.logger.record("reward/level", info["reward_level"])
#                 self.logger.record("reward/badge", info["reward_badge"])
#                 self.logger.record("reward/event", info["reward_event"])
#                 self.logger.record("reward/exp", info["reward_exp"])
#                 self.logger.record("reward/heal", info["reward_heal"])
#                 self.logger.record("reward/dmg", info["reward_dmg"])
            
#         return True
class GameStatsCallback(BaseCallback):
    """
    12개 환경의 정보를 모두 모아서 '평균(Mean)'을 기록하는 수정된 콜백
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        # 1. 데이터를 담을 빈 리스트 준비
        total_rewards = []
        badges = []
        level_sums = []
        explores = []
        deaths = []
        
        # 보상 세부 항목 리스트
        r_explore = []
        r_level = []
        r_badge = []
        r_event = []
        r_exp = []
        r_heal = []
        r_dmg = []
        r_stuck = []

        # 2. 12개 환경을 순회하며 데이터 수집
        for info in infos:
            if "reward_total" in info:
                total_rewards.append(info["reward_total"])
                badges.append(info["stats_badges"])
                level_sums.append(info["stats_level_sum"])
                explores.append(info["stats_explore"])
                deaths.append(info["stats_deaths"])
                
                r_explore.append(info["reward_explore"])
                r_level.append(info["reward_level"])
                r_badge.append(info["reward_badge"])
                r_event.append(info["reward_event"])
                r_exp.append(info["reward_exp"])
                r_heal.append(info["reward_heal"])
                r_dmg.append(info["reward_dmg"])
                r_stuck.append(info["reward_stuck"])

        # 3. 데이터가 모였으면 '평균'을 내서 기록 (np.mean 사용)
        if total_rewards:
            self.logger.record("game/badges", np.mean(badges))
            self.logger.record("game/level_sum", np.mean(level_sums))
            self.logger.record("game/exploration", np.mean(explores))
            self.logger.record("game/deaths", np.mean(deaths))
            
            self.logger.record("reward/total", np.mean(total_rewards))
            self.logger.record("reward/exploration", np.mean(r_explore))
            self.logger.record("reward/level", np.mean(r_level))
            self.logger.record("reward/badge", np.mean(r_badge))
            self.logger.record("reward/event", np.mean(r_event))
            self.logger.record("reward/exp", np.mean(r_exp))
            self.logger.record("reward/heal", np.mean(r_heal))
            self.logger.record("reward/dmg", np.mean(r_dmg))
            self.logger.record("reward/stuck", np.mean(r_stuck))

            self.logger.record("game/badges_max", np.max(badges))
            self.logger.record("game/level_sum_max", np.max(level_sums))
            self.logger.record("game/exploration_max", np.max(explores)) # 가장 멀리 간 놈!
            self.logger.record("reward/event_max", np.max(r_event))
            self.logger.record("reward/total_max", np.max(total_rewards)) # 점수 제일 높은 놈
            self.logger.record("reward/stuck_min", np.min(r_stuck)) # 제일 stuck 패널티 많이 받은 놈!
            
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
        
        # [설정] 녹화할 길이 (프레임 수)
        self.record_length = 100000
        self.record_config['max_steps'] = self.record_length + 1000 

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            print(f"\n🎥 [Video] 녹화 시작... (Step: {self.num_timesteps})")
            try:
                # 녹화용 환경 생성
                eval_env = GoldEnv(self.record_config)
                obs, _ = eval_env.reset()
                done = False
                truncated = False
                step_cnt = 0
                
                # 저장 경로 설정
                rollout_dir = os.path.join(self.sess_path, "rollouts")
                os.makedirs(rollout_dir, exist_ok=True)
                
                # [수정 1] 윈도우 호환성을 위해 .avi 확장자 사용
                video_path = os.path.join(rollout_dir, f"video_step_{self.num_timesteps}.avi")

                # -----------------------------------------------------------
                # [WinError 87 해결] OpenCV (cv2) 사용 코드로 교체됨
                # -----------------------------------------------------------
                
                # 1. 첫 프레임 가져와서 해상도 자동 감지
                dummy_screen = eval_env.pyboy.screen.ndarray[:, :, :3]
                height, width, channels = dummy_screen.shape
                
                # 2. 코덱 설정 (MJPG가 윈도우에서 가장 에러가 안 남)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                
                # 3. Writer 생성 (FPS 60)
                writer = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))

                if not writer.isOpened():
                    print("❌ [Video] 녹화 파일을 열 수 없습니다.")
                    eval_env.close()
                    return True

                # 4. 녹화 루프
                while not (done or truncated) and step_cnt < self.record_length:
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, done, truncated, _ = eval_env.step(action)
                    
                    # 화면 가져오기
                    raw_screen = eval_env.pyboy.screen.ndarray[:, :, :3]
                    
                    # [중요] RGB -> BGR 변환 (OpenCV 색상 순서 맞추기)
                    frame_bgr = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
                    
                    # 쓰기
                    writer.write(frame_bgr)
                    step_cnt += 1
                    
                # 리소스 해제
                writer.release()
                eval_env.close()
                print(f"✅ [Video] 저장 완료: {video_path}")
            
            except Exception as e:
                print(f"❌ [Video] 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                
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
            
            # [수정된 부분] 중요: features_extractor(시각 정보 처리)만 가져옵니다!
            # 행동 결정 레이어(mlp_extractor, action_net)는 가져오지 않고 초기화 상태로 둡니다.
            for key in new_model_dict.keys():
                if key in state_dict and "features_extractor" in key:
                    if new_model_dict[key].shape == state_dict[key].shape:
                        new_model_dict[key] = state_dict[key]
                        copied_count += 1
            
            new_model.policy.load_state_dict(new_model_dict)
            print(f">>> 로드 완료: {copied_count}개 레이어 복사됨 (Feature Extractor Only).")
            
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
        unfreeze_step = 0
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
        'max_steps': 40960 * 24,
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
        learning_rate=0.00008, # [최적화] 학습률 8e-05
        n_steps=4096, # ----------------------------------------------------- 로그 찍히는거 빨리 하려고 수정함 원래 4096
        batch_size=128, 
        n_epochs=10, 
        gamma=0.997, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        ent_coef=0.02, # 0.02
    )

    # if target_weights:
    #     model = load_weights_from_zip(model, target_weights)
    #     if not is_resume:
    #         for name, param in model.policy.named_parameters():
    #             if 'features_extractor' in name: param.requires_grad = False
    if target_weights:
        if is_resume:
            # ✅ 이어하기: SB3 공식 load 함수 사용 (뇌 + 눈 + 학습 상태 모든 것 복구)
            print(f"🔄 [이어하기] 모델을 통째로 복구합니다... ({target_weights})")
            
            # 주의: PPO.load는 새로운 모델 객체를 반환하므로 model 변수에 다시 담아야 합니다.
            # custom_objects는 혹시 모를 버전 호환성 등을 위해 필요할 수 있습니다.
            #model = PPO.load(target_weights, env=env) #----------------------------------------- 로그 찍히는거 빨리 하려고 수정함//
            model = PPO.load(target_weights, env=env, n_steps=4096, batch_size=128, learning_rate=0.00025, ent_coef=0.03) # 애가 너무 멍청해서 learning_rate=0.00008 에서 0.00025로 수정해봄 ent_coef는 0.02에서 0.03.
            
        else:
            # ✅ 전이학습 (Red -> Gold): 사용자 정의 함수 사용 (눈만 이식)
            print(f"🆕 [전이학습] Feature Extractor(눈)만 이식합니다...")
            model = load_weights_from_zip(model, target_weights)
            
            # 전이학습일 때만 눈을 얼림 (Freeze)
            for name, param in model.policy.named_parameters():
                if 'features_extractor' in name: param.requires_grad = False

    target_global_step = 100000
    save_freq = target_global_step // num_cpu
    print(f"💾 저장 및 녹화 주기: 전체 {target_global_step} 스텝 (CPU당 {save_freq} 스텝)")
    
    # ... (기존 콜백 리스트 정의 부분) ...
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix=f'gold_{sess_id}'),
        # VideoRecorderCallback(env_config, save_freq=save_freq, log_dir=log_dir), # ------------------------------------------- 영상 녹화 기능. 빠른 학습을 위해 일단 꺼둠.
        
        # [추가] 우리가 만든 게임 스탯 로거
        GameStatsCallback(), 
        
        UnfreezeCallback(unfreeze_step=unfreeze_step)
    ])
    
    # model.learn(...)

    print(f"\n🚀 학습 시작! (Session: {sess_id})")
    model.learn(total_timesteps=50_000_000, callback=callbacks)
    model.save(f"{sess_path}/final_model")
    env.close()