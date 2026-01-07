import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import config


class MetricLoggingCallback(BaseCallback):
    """
    사용자 정의 지표(탐험 수, 레벨 합, 보상 상세)를 Tensorboard 및 콘솔 표에 기록하는 콜백
    """

    def __init__(self, verbose=0):
        super(MetricLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # 1. 기본 통계 추출
        badges = [info.get("badges", 0) for info in infos]
        explorations = [info.get("exploration", 0) for info in infos]
        level_sums = [info.get("level_sum", 0) for info in infos]
        deaths = [info.get("deaths", 0) for info in infos]
        heals = [info.get("heal", 0) for info in infos]  # 추가

        # 2. 보상 상세 내역 추출 (GetReward에서 넘겨준 dict)
        rew_badge = [info.get("reward_details", {}).get("badge", 0) for info in infos]
        rew_gemini = [info.get("reward_details", {}).get("gemini", 0) for info in infos]

        # [추가] 새로운 보상 로그
        rew_battle = [info.get("reward_details", {}).get("battle", 0) for info in infos]
        rew_exp = [info.get("reward_details", {}).get("exp", 0) for info in infos]
        rew_dmg = [info.get("reward_details", {}).get("dmg", 0) for info in infos]
        rew_dead = [info.get("reward_details", {}).get("dead", 0) for info in infos]

        # 3. 로거에 기록
        self.logger.record("game/badges", np.mean(badges))
        self.logger.record("game/exploration", np.mean(explorations))
        self.logger.record("game/level_sum", np.mean(level_sums))
        self.logger.record("game/deaths", np.mean(deaths))
        self.logger.record("game/heals", np.mean(heals))

        # 리워드 그래프 (학습 경향 확인용)
        self.logger.record("reward/badge", np.mean(rew_badge))
        self.logger.record("reward/gemini", np.mean(rew_gemini))
        self.logger.record("reward/battle", np.mean(rew_battle))  # 전투 승리
        self.logger.record("reward/exp", np.mean(rew_exp))  # 경험치 획득
        self.logger.record("reward/dmg", np.mean(rew_dmg))  # 딜링
        self.logger.record("reward/dead", np.mean(rew_dead))  # 기절 패널티

        return True

class SpeedrunCallback(BaseCallback):
    """
    각 배지 도달 시 '최소 스텝'을 갱신했을 때만 모델을 저장하는 콜백
    """

    def __init__(self, save_path: str, verbose=1):
        super(SpeedrunCallback, self).__init__(verbose)
        self.save_path = save_path
        self.record_file = os.path.join(save_path, "speedrun_records.json")

        # [0 ~ 16] 배지별 최소 스텝 기록 (초기값은 무한대)
        # 파일이 있으면 불러오고, 없으면 새로 만듭니다.
        if os.path.exists(self.record_file):
            with open(self.record_file, "r") as f:
                self.best_steps = {int(k): v for k, v in json.load(f).items()}
            if verbose > 0:
                print(f"📖 [Speedrun] 기존 기록을 불러왔습니다: {self.best_steps}")
        else:
            self.best_steps = {i: float('inf') for i in range(17)}  # 배지 0~16개

    def _save_records(self):
        """현재 최고 기록을 JSON 파일로 저장"""
        with open(self.record_file, "w") as f:
            json.dump(self.best_steps, f, indent=4)

    def _on_step(self) -> bool:
        # VecEnv에서는 infos가 리스트로 옵니다.
        infos = self.locals.get("infos", [])

        for idx, info in enumerate(infos):
            # 환경에서 '현재 배지 수'와 '현재 에피소드 스텝 수'를 가져옴
            # 주의: GoldEnv의 info에 'step_count'가 있어야 합니다!
            current_badge = info.get("badges", 0)
            current_step = info.get("step_count", 0)

            # 배지가 0개일 때는 굳이 저장 안 함 (원하면 포함 가능)
            if current_badge > 0:
                # 🏆 신기록 달성 체크 (작을수록 좋음)
                if current_step < self.best_steps.get(current_badge, float('inf')):

                    old_record = self.best_steps.get(current_badge, float('inf'))
                    self.best_steps[current_badge] = current_step  # 기록 갱신
                    self._save_records()  # 파일에도 기록

                    if self.verbose > 0:
                        diff = old_record - current_step if old_record != float('inf') else 0
                        print(f"⚡ [NEW RECORD] 배지 {current_badge}개 달성! (Env {idx})")
                        print(f"   ㄴ 기존: {old_record} -> 신규: {current_step} steps (단축: {diff})")

                    # 모델 저장 (파일명에 스텝 수 포함)
                    # 예: best_badge_1_step_5020.zip
                    model_name = f"best_badge_{current_badge}_step_{current_step}"
                    save_path = os.path.join(self.save_path, model_name)
                    self.model.save(save_path)

                    if self.verbose > 0:
                        print(f"   💾 모델 저장 완료: {model_name}.zip")

        return True


def get_callbacks(checkpoint_dir):
    #10000스텝마다, rl_model 정책 저장.
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="rl_model"
    )

    speedrun_callback = SpeedrunCallback(
        save_path=config.MODELS_DIR,
        verbose=1
    )

    #텐서보드용 로깅 롤백
    logging_callback = MetricLoggingCallback()
    
    #리스트로 묶어서 반환 --> main.py의 model.learn에 줌.
    return CallbackList([checkpoint_callback, speedrun_callback, logging_callback])