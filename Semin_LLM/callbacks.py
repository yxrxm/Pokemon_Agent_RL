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
        # VecEnv를 사용하므로 infos는 여러 환경의 리스트입니다.
        infos = self.locals.get("infos", [])

        if not infos:
            return True

        # 1. 값 추출 (병렬 환경이 여러 개일 경우 통계 냄)
        badges = [info.get("badges", 0) for info in infos]
        explorations = [info.get("exploration", 0) for info in infos]
        level_sums = [info.get("level_sum", 0) for info in infos]
        deaths = [info.get("deaths", 0) for info in infos]

        # 보상 상세 내역 추출
        rew_badge = [info.get("reward_details", {}).get("badge", 0) for info in infos]
        rew_gemini = [info.get("reward_details", {}).get("gemini", 0) for info in infos]

        # 2. 로거에 기록 (SB3가 알아서 표로 만들어줍니다)
        # [Game Stats]
        self.logger.record("game/badges", np.mean(badges))
        self.logger.record("game/badges_max", np.max(badges))
        self.logger.record("game/exploration", np.mean(explorations))
        self.logger.record("game/exploration_max", np.max(explorations))
        self.logger.record("game/level_sum", np.mean(level_sums))
        self.logger.record("game/level_sum_max", np.max(level_sums))
        self.logger.record("game/deaths", np.mean(deaths))

        # [Reward Breakdown]
        self.logger.record("reward/badge", np.mean(rew_badge))
        self.logger.record("reward/event", np.mean(rew_gemini))  # Gemini 점수

        return True


class BadgeSaveCallback(BaseCallback):
    """
    배지 개수가 늘어날 때마다 모델을 별도로 저장하는 커리큘럼 학습용 콜백
    """

    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(BadgeSaveCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.last_max_badges = 0

    def _on_training_start(self) -> None:
        badges = self.training_env.get_attr("current_badge_count")
        self.last_max_badges = max(badges) if badges else 0
        if self.verbose > 0:
            print(f"👀 [Callback] 현재 최대 배지 수: {self.last_max_badges}개로 감시를 시작합니다.")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            current_badges_list = self.training_env.get_attr("current_badge_count")
            current_max = max(current_badges_list)

            if current_max > self.last_max_badges:
                if self.verbose > 0:
                    print(f"🎉 [성장] 배지 {current_max}개를 획득한 환경이 있습니다!")

                # 이전 단계 졸업 저장
                prev_model_name = f"final_model_badge_{self.last_max_badges}"
                path = os.path.join(self.save_path, prev_model_name)
                self.model.save(path)

                # 기준 업데이트
                self.last_max_badges = current_max

                # 새로운 단계 시작 저장
                new_model_name = f"final_model_badge_{current_max}"
                path_new = os.path.join(self.save_path, new_model_name)
                self.model.save(path_new)

                if self.verbose > 0:
                    print(f"💾 [저장] {new_model_name}.zip 생성됨")
        return True


def get_callbacks(checkpoint_dir):
    # 3가지 콜백 합치기: 체크포인트 + 배지저장 + 로그표시
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="rl_model"
    )

    badge_callback = BadgeSaveCallback(
        check_freq=1000,
        save_path=config.MODELS_DIR
    )

    logging_callback = MetricLoggingCallback()

    return CallbackList([checkpoint_callback, badge_callback, logging_callback])