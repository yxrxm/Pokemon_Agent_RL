import os
#config 변수 제어용 파일.

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #현재 파일의 위치의 경로를 따옴
MODELS_DIR = os.path.join(BASE_DIR, "models") #models 파일은 ai 학습을 통한 정책(policy) 파일이고 그 파일에서 정책을 끌어다 사용하는 것이라 위치 경로를 설정함.
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
#sessions 파일은 각 게임을 실행했을 때 저장되는 파일이고 여기에도 정책 파일이 저장됨.
#필요하거나 하고싶으면 models 파일에 final_model_badge_0, 1, 2... 이런 식으로 끌어올 정책 수정가능
GB_PATH = os.path.join(BASE_DIR, "game_file/PokeGold.gbc") #BASE_DIR 기준으로 game_file 속의 게임 파일의 경로
INIT_STATE_FILE = os.path.join(BASE_DIR, "game_file/init.state")
#BASE_DIR을 기준으로 init.state 의 경로.
#init.state는 pyboy 라이브러리를 통해서 만든 것이고 게임boy 파일로는 연동되지 않음.

'''실제 변수들'''
#보상변수의 가중치(PPO)를 설정할 수 있는 부분 // 뱃지의 개수에 따라 다른 가중치를 사용하기로 함. !!!
REWARD_WEIGHTS = {
    "default": { "exploration": 0.1, "event": 1.0, "battle": 5.0, "level": 5.0, "gemini": 1.0, "exp": 20.0, "dmg": 300.0, "heal": 10.0, "dead": 0.1 }
}

#기본 환경 변수
env_config = {
    "save_video": True,
    "headless": True, #pyboy 시뮬레이터를 보이게 할 지의 여부
    "action_freq": 24, #에이전트가 버튼을 한 번 누르면 몇 프레임 동안 꾹 누르고 있을까의 여부, 보통 24프레임 -> 0.4초를 활용함.
    "max_steps": 2048 * 100, #main.py를 실행했을 때, 최대 에이전트의 step 수
    "gb_path": GB_PATH,
    "init_state": INIT_STATE_FILE,
}

#PPO 정책 변수
train_config = {
    "total_timesteps": 50_000_000, #AI가 버튼을 누르는 횟수
    "learning_rate": 0.0003, #정책 weight를 변화시키는 비율, 너무 크면 확 변하고 너무 작으면 오래 걸림. PPO에서는 일반적으로 0.0001~0.0003을 사용함.
    "n_steps": 2048, #가중치 Update까지의 스텝 수, 2048까지 정보를 모음.
    "batch_size":  128, #64개의 데이터를 가지고 2048 즉, 2048 / 64 -> 32번 학습을 업데이트 함. 너무 작으면 오래 걸림.
    "n_envs": 2, #멀티프로세싱 개수, 동시에 몇 개의 게임을 돌릴 지.
}

#LLM 환경 변수
ai_config = {
    "use_ai_coach": True, #AI의 사용 여부, PPO를 도우기 위한 것이고 RL이 잘 되면 끄고 해도 됨.
    "project_id": "", # 프로젝트 ID // 사용할 AI의 실제 ID
    "location": "us-central1", #미국 서버
    "model_name": "models/gemini-2.0-flash-001",
    # model_name 인데, gemini 에서도 사용할 수 있는 model이 다르고 비용도 다름.
    "coach_interval": 512, #LLM을 몇 스텝마다 호출할 지.
    "key_path": "" #실제 AI 사용 키 json 파일
}
#LLM을 사용하는 것은 PPO만으로는 시간이 너무 걸리기 때문에, 그 시간을 줄이기 위함임.
#정답을 풀도록 이끌어 주는 선생님 느낌이라고 생각하면 됨.