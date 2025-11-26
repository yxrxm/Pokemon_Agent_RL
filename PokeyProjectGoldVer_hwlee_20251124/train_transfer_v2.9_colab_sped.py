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

# [추가] 텐서보드 프로그램 모듈 (스크립트 내 실행용)
from tensorboard import program

# 사용자가 가지고 있는 환경 파일 임포트
from GoldEnv import GoldEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from tensorboard_callback import TensorboardCallback

# ==================================================================================
# [1] 환경 생성 함수
# ==================================================================================
def make_env(rank, env_conf, seed=0):
    """
    멀티프로세싱 환경 생성을 위한 유틸리티 함수
    :param rank: 프로세스 인덱스 (0, 1, 2...)
    :param env_conf: 환경 설정 딕셔너리
    :param seed: 랜덤 시드
    """
    def _init():
        env = GoldEnv(env_conf)
        # 각 프로세스마다 서로 다른 시드를 주어 다양성 확보
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

# ==================================================================================
# [2] 최신 체크포인트 자동 검색 함수
# ==================================================================================
def find_latest_checkpoint(log_dir):
    """
    runs 폴더에서 가장 최근에 생성된 .zip 파일을 찾습니다.
    이어하기(Resume)나 전이학습(Transfer Learning) 시 사용됩니다.
    """
    # runs 폴더 내의 모든 zip 파일 검색
    list_of_files = glob.glob(os.path.join(log_dir, "*.zip"))
    if not list_of_files:
        return None
    # 생성 시간(getctime) 순으로 정렬하여 가장 마지막 파일 반환
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

# ==================================================================================
# [3] Real Color 영상 녹화 콜백 (RAM 최적화: 스트리밍 저장 방식)
# ==================================================================================
# class VideoRecorderCallback(BaseCallback):
#     def __init__(self, eval_env_config, save_freq, log_dir, verbose=0):
#         super().__init__(verbose)
#         self.save_freq = save_freq
#         self.log_dir = log_dir
#         self.sess_path = eval_env_config['session_path']
        
#         # 녹화용 환경 설정 복사 (기본 학습 환경과 분리)
#         self.record_config = eval_env_config.copy()
#         self.record_config['headless'] = True
        
#         # [중요] headless가 True여도 save_video=True로 해야 
#         # PyBoy가 내부적으로 화면 버퍼를 업데이트합니다. (False면 검은 화면만 나옴)
#         self.record_config['save_video'] = True 
#         self.record_config['instance_id'] = 'recorder'

#         # [설정] 녹화할 길이 (스텝 수)
#         # 10만 스텝을 전부 찍어서 전체 진행 과정을 봅니다.
#         self.record_length = 100000 
        
#         # 환경의 최대 스텝은 녹화 길이보다 조금 더 넉넉하게 잡아 
#         # 녹화 도중 강제 종료되는 것을 방지합니다.
#         self.record_config['max_steps'] = self.record_length + 1000 

#     def _on_step(self) -> bool:
#         # 저장 주기(save_freq)가 되었을 때 실행
#         if self.n_calls % self.save_freq == 0:
#             print(f"\n🎥 [Video] {self.num_timesteps} 스텝: 영상 녹화 시작 ({self.record_length} frames)...")
#             print("   -> RAM 최적화 모드: 프레임을 메모리에 쌓지 않고 즉시 파일로 기록합니다.")
            
#             try:
#                 # 별도의 평가용 환경 생성
#                 eval_env = GoldEnv(self.record_config)
#                 obs, _ = eval_env.reset()
                
#                 done = False
#                 truncated = False
#                 step_cnt = 0
                
#                 # 영상 저장 경로 설정 (session_path/rollouts 폴더)
#                 rollout_dir = os.path.join(self.sess_path, "rollouts")
#                 os.makedirs(rollout_dir, exist_ok=True)
                
#                 video_path = os.path.join(rollout_dir, f"video_step_{self.num_timesteps}.mp4")

#                 # [RAM 최적화 핵심]
#                 # 기존 방식(list append)은 10만 장을 메모리에 다 올리므로 Colab RAM(12GB)이 터집니다.
#                 # VideoWriter를 컨텍스트 매니저(with)로 열어서, 이미지를 한 장 받자마자 파일에 쓰고 메모리는 비웁니다.
#                 # fps=1200 설정: 10만 프레임을 약 1분 20초 내외로 압축하여 빠르게 훑어보기 위함 (20배속)
#                 with media.VideoWriter(video_path, shape=(144, 160), fps=1200) as writer:
#                     while not (done or truncated) and step_cnt < self.record_length:
#                         # 현재 모델 정책으로 행동 예측
#                         action, _ = self.model.predict(obs, deterministic=False)
#                         obs, _, done, truncated, _ = eval_env.step(action)
                        
#                         # [핵심] PyBoy 원본 화면(RGB) 가져오기
#                         # GoldEnv.render()는 흑백 축소판을 반환하므로, pyboy 객체에서 직접 가져옵니다.
#                         raw_screen = eval_env.pyboy.screen.ndarray[:, :, :3]
                        
#                         # [수정] 리스트에 담지 않고(append X), 파일 스트림에 바로 씀(add_image O)
#                         writer.add_image(raw_screen)
#                         step_cnt += 1
                        
#                         # 진행 상황 표시 (너무 오래 걸리니 1만 스텝마다 로그 출력)
#                         if step_cnt % 10000 == 0:
#                             print(f"   -> 녹화 진행 중... ({step_cnt}/{self.record_length})")
                
#                 eval_env.close()
#                 print(f"✅ [Video] 저장 완료: {video_path} (길이: {step_cnt} frames, FPS: 1200)")
                
#             except Exception as e:
#                 print(f"❌ [Video] 저장 중 오류 발생: {e}")
            
#         return True

# ==================================================================================
# [4] Transfer Learning 관련 함수 (Unfreeze)
# ==================================================================================
class UnfreezeCallback(BaseCallback):
    def __init__(self, unfreeze_step=500_000, verbose=0):
        super().__init__(verbose)
        self.unfreeze_step = unfreeze_step
        self.is_frozen = True

    def _on_step(self) -> bool:
        # 지정된 스텝 수에 도달하면 얼려뒀던 레이어를 녹임
        if self.is_frozen and self.num_timesteps >= self.unfreeze_step:
            print(f"\n🔥 [Unfreeze] {self.num_timesteps} 스텝 도달! Feature Extractor(CNN)를 녹입니다.")
            for param in self.model.policy.parameters():
                param.requires_grad = True
            # 미세 조정을 위해 학습률을 낮춤
            self.model.learning_rate = 0.0001 
            self.is_frozen = False
        return True

def load_weights_from_zip(new_model, file_path, device="cpu"):
    """
    .zip 파일이나 .pth 파일에서 모델 가중치를 로드합니다.
    가중치 딕셔너리의 키(Key)와 형태(Shape)가 맞는 것만 골라서 복사합니다.
    (전이 학습 시 모델 구조가 약간 달라도 유연하게 대처 가능)
    """
    print(f"\n=== 가중치 로드 시도: {file_path} ===")
    state_dict = None
    try:
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, 'r') as archive:
                # SB3 저장 방식: zip 안에 policy.pth가 들어있음
                if 'policy.pth' in archive.namelist():
                    print("   -> Zip 내부의 'policy.pth'를 발견했습니다.")
                    with archive.open('policy.pth') as f:
                        buffer = io.BytesIO(f.read())
                        state_dict = torch.load(buffer, map_location=device)
                else:
                    print("   -> Zip 내부에 policy.pth가 없습니다. 전체 모델 로드를 시도합니다.")
                    return PPO.load(file_path, env=new_model.get_env())
        else:
            # 일반 pth 파일일 경우
            state_dict = torch.load(file_path, map_location=device)

        if state_dict is not None:
            # 딕셔너리 형태 처리 (OrderedDict 등)
            if not isinstance(state_dict, dict) and hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            
            new_model_dict = new_model.policy.state_dict()
            copied_count = 0
            
            # 레이어 이름과 크기가 일치하는 것만 복사
            for key in new_model_dict.keys():
                if key in state_dict and new_model_dict[key].shape == state_dict[key].shape:
                    new_model_dict[key] = state_dict[key]
                    copied_count += 1
            
            new_model.policy.load_state_dict(new_model_dict)
            print(f">>> 로드 완료: {copied_count}개 레이어 복사됨.")
            
    except Exception as e:
        print(f"[오류] 가중치 로드 실패: {e}")
        print(">>> 새로운 모델로 시작합니다.")
        
    return new_model

# ==================================================================================
# [5] 텐서보드 자동 실행 함수
# ==================================================================================
def launch_tensorboard(log_dir):
    """
    코드 실행 시 백그라운드에서 TensorBoard 서버를 엽니다.
    """
    try:
        tb = program.TensorBoard()
        # 포트 6006으로 열고, 모든 인터페이스(bind_all)에서 접속 허용
        tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006', '--bind_all'])
        url = tb.launch()
        print(f"\n📊 [TensorBoard] 서버가 시작되었습니다: {url}")
        print("   -> Colab 사용자는 아래의 매직 커맨드를 별도 셀에서 실행하는 것이 더 좋습니다:")
        print(f"      %load_ext tensorboard")
        print(f"      %tensorboard --logdir {log_dir}")
    except Exception as e:
        print(f"\n⚠️ [TensorBoard] 자동 실행 실패 (수동 실행 권장): {e}")

# ==================================================================================
# [6] 메인 함수
# ==================================================================================
if __name__ == '__main__':
    # 1. 세션 ID 생성 (폴더 구분용)
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f'session_gold_{sess_id}')
    sess_path.mkdir(exist_ok=True)
    
    log_dir = "./runs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 2. init.state 확인 (필수)
    if not os.path.exists("./init.state"):
        print("⚠️ [경고] 'init.state' 파일이 없습니다!")
        print("   -> AI가 오박사 설명 듣느라 시간을 다 쓸 수 있습니다.")
        
    # [추가] 학습 시작 전에 텐서보드 백그라운드 실행 시도
    launch_tensorboard(log_dir)

    # 3. 체크포인트 확인 및 로드 설정
    target_weights = find_latest_checkpoint(log_dir)
    is_resume = False
    
    # 기존 학습 파일이 없으면 Red 버전 베이스 모델(전이 학습용) 확인
    base_red_path = "./poke_26214400/policy.pth" 

    if target_weights:
        print(f"🔄 [이어하기] 최신 체크포인트 발견: {target_weights}")
        is_resume = True
        unfreeze_step = 100_000 # 이어하기면 10만 스텝 후에 풂
    elif os.path.exists(base_red_path):
        print(f"🆕 [전이학습] Red 버전 가중치 로드: {base_red_path}")
        target_weights = base_red_path
        unfreeze_step = 500_000 # 전이학습이면 50만 스텝 동안 적응시킴
    else:
        print("🆕 [새로하기] 가중치 없이 처음부터 학습합니다.")
        target_weights = None
        unfreeze_step = 0

    # 4. 환경 설정
    env_config = {
        'headless': True,            # 학습 속도를 위해 화면 렌더링 끔
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,           # 행동 반복 횟수
        'init_state': './init.state', # 시작 세이브 파일
        'max_steps': 2048 * 10,       # 단일 에피소드 최대 길이 (학습용)
        'print_rewards': True,
        'save_video': False,          # 학습 중엔 자동 저장 끔 (콜백이 따로 처리)
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': './PokeGold.gbc',
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }

    # 5. 병렬 환경 생성 (CPU 개수 조절)
    num_cpu = min(os.cpu_count(), 8) 
    print(f"⚙️ 설정된 병렬 환경 개수: {num_cpu}")
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # 6. PPO 모델 생성
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

    # 7. 가중치 로드 및 Freeze 설정
    if target_weights:
        model = load_weights_from_zip(model, target_weights)
        
        # 전이학습인 경우에만 초반에 시각 처리 부분(CNN)을 얼림
        if not is_resume:
            print("🥶 CNN(시각 처리) 레이어를 얼립니다.")
            for name, param in model.policy.named_parameters():
                if 'features_extractor' in name:
                    param.requires_grad = False

    # 8. 콜백 설정
    # 전체 10만 스텝마다 저장 및 녹화를 수행
    # SubprocVecEnv 사용 시, 업데이트 주기는 (n_steps * num_cpu) 기준이므로
    # 전체 스텝 수를 CPU 개수로 나눠주어야 정확한 타이밍을 잡음
    target_global_step = 100000
    save_freq = target_global_step // num_cpu
    print(f"💾 저장 및 녹화 주기: 전체 {target_global_step} 스텝 (CPU당 {save_freq} 스텝)")
    
    callbacks = CallbackList([
        CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix=f'gold_{sess_id}'),
        # VideoRecorderCallback(env_config, save_freq=save_freq, log_dir=log_dir),
        TensorboardCallback(log_dir=log_dir),
        UnfreezeCallback(unfreeze_step=unfreeze_step)
    ])

    # 9. 학습 시작
    print(f"\n🚀 학습 시작! (Session: {sess_id})")
    print("   - Tensorboard 실행: tensorboard --logdir=runs")
    print("   - 종료하려면 Ctrl+C를 누르세요.")
    
    try:
        model.learn(total_timesteps=50_000_000, callback=callbacks)
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 학습 중단됨.")
        
    # 저장 및 종료
    model.save(f"{sess_path}/final_model")
    env.close()
    print("✅ 학습 종료 및 저장 완료.")