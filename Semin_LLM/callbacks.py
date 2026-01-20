import os
import numpy as np
import json
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
        #전체 보상
        total_reward = [info.get("stats/total_reward") for info in infos]

        #Game 내 count 내역
        badges = [info.get("game/badges", 0) for info in infos]
        explorations = [info.get("game/exploration", 0) for info in infos]
        map_count = [info.get("game/map_count", 0) for info in infos]
        level_sums = [info.get("game/level_sum", 0) for info in infos]
        deaths = [info.get("game/deaths", 0) for info in infos]
        heals_battle = [info.get("game/heal_battle", 0) for info in infos]
        heals_field = [info.get("game/heal_field", 0) for info in infos]
        wins = [info.get("game/battle_wins", 0) for info in infos]
        opp_max_level = [info.get("game/opp_max_level", 0) for info in infos]
        party_max_levels = [info.get("game/party_max_level", 0) for info in infos]
        overlap_step = [info.get("game/overlap_step", 0) for info in infos]
        money = [info.get("game/money", 0) for info in infos]
        battle_penalty = [info.get("reward/battle_penalty", 0) for info in infos]
        trainer_Bcount = [info.get("game/battle_trainer", 0) for info in infos]

        #보상 상세 내역
        rew_badge = [info.get("reward/badge", 0) for info in infos]
        rew_battle = [info.get("reward/battle", 0) for info in infos]
        rew_exp = [info.get("reward/exp", 0) for info in infos]
        rew_dmg = [info.get("reward/dmg", 0) for info in infos]
        rew_dead = [info.get("reward/dead", 0) for info in infos]
        rew_explore = [info.get("reward/explore", 0) for info in infos]
        rew_penalty = [info.get("reward/penalty", 0) for info in infos]
        rew_stuck = [info.get("reward/stuck", 0) for info in infos]
        rew_event = [info.get("reward/event", 0) for info in infos]
        rew_level = [info.get("reward/level", 0) for info in infos]
        rew_gemini = [info.get("reward/gemini", 0) for info in infos]
        rew_map = [info.get("reward/map", 0) for info in infos]

        
        self.logger.record("스텟/전체 보상", np.mean(total_reward))
        #게임 내 횟수 내역
        self.logger.record("게임/뱃지 개수", np.mean(badges))
        self.logger.record("게임/탐험 좌표 수", np.mean(explorations))
        self.logger.record("게임/탐험 맵 수", np.mean(map_count))
        self.logger.record("게임/내 파티 레벨 합", np.mean(level_sums))
        self.logger.record("게임/내 파티 최대 레벨", np.mean(party_max_levels))
        self.logger.record("게임/사망 횟수", np.mean(deaths))
        self.logger.record("게임/배틀 힐 횟수", np.mean(heals_battle))
        self.logger.record("게임/필드 힐 횟수", np.mean(heals_field))
        self.logger.record("게임/전투 승리 횟수", np.mean(wins))
        self.logger.record("게임/상대 최대 레벨", np.max(opp_max_level))
        self.logger.record("게임/중복 스텝 횟수 (100 중 20회 이상일 경우)", np.mean(overlap_step))
        self.logger.record("게임/소지한 돈", np.mean(money))
        self.logger.record("게임/트레이너 배틀 수", np.mean(trainer_Bcount))

        #보상 상세 내역
        self.logger.record("보상/뱃지개수", np.mean(rew_badge))
        self.logger.record("보상/승리 보상", np.mean(rew_battle))
        self.logger.record("보상/경험치 보상", np.mean(rew_exp))
        self.logger.record("보상/대미지 보상", np.mean(rew_dmg))
        self.logger.record("보상/탐험 보상", np.mean(rew_explore))
        self.logger.record("보상/사망 패널티", np.mean(rew_dead))
        self.logger.record("보상/중복 스텝 패널티", np.mean(rew_penalty))
        self.logger.record("보상/벽 박기 패널티", np.mean(rew_stuck))
        self.logger.record("보상/배틀 지속 패널티", np.mean(battle_penalty))
        self.logger.record("보상/제미나이 보상", np.mean(rew_gemini))
        self.logger.record("보상/이벤트 보상", np.mean(rew_event))
        self.logger.record("보상/레벨업 보상", np.mean(rew_level))
        self.logger.record("보상/새로운 맵 보상", np.mean(rew_map))

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
            current_badge = info.get("game/badges", 0)
            current_step = info.get("stats/step_count", 0)

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
                        print(f"모델 저장 완료: {model_name}.zip")

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