import numpy as np

class BattleState:
    def __init__(self, env):
        self.env = env

    def get_battle_vector(self):
        if not self.env.is_in_battle():
            return np.zeros(11, dtype=np.float32)
        """
        정리해주신 RAM 주소를 사용하여 10차원의 전투 벡터를 생성합니다.
        결과값은 모두 0~1 사이로 정규화하여 AI가 학습하기 좋게 만듭니다.
        """
        # 1. 내 포켓몬 HP 비율 ($CB1C: 현재, $DA4E: 최대)
        my_hp = self.env.read_hp(0xCB1C)
        my_max_hp = self.env.read_hp(0xDA4E)
        my_hp_pct = my_hp / max(1, my_max_hp)

        # 2. 상대 포켓몬 HP 비율 ($D0FF: 현재, $D101: 최대)
        en_hp = self.env.read_hp(0xD0FF)
        en_max_hp = self.env.read_hp(0xD101)
        en_hp_pct = en_hp / max(1, en_max_hp)
        

        # 3. 레벨 비교 ($DA49: 내 레벨, $D0FC: 상대 레벨)
        my_lvl = self.env.read_m(0xDA49)
        en_lvl = self.env.read_m(0xD0FC)
        lvl_ratio = my_lvl / (my_lvl + en_lvl + 1e-5)

        # 4. 내 기술 정보 ($CB0E ~ $CB11) - 기술 존재 여부만 체크 (0 or 1)
        # 기술 ID를 그대로 넣으면 숫자가 너무 커서 AI가 헷갈려하므로 존재 여부만 우선 체크
        moves = [self.env.read_m(addr) for addr in [0xCB0E, 0xCB0F, 0xCB10, 0xCB11]]
        has_moves = [1.0 if m > 0 else 0.0 for m in moves]

        # 5. 상태 이상 ($CB1A: 나, $D0FD: 상대) - 있으면 1, 없으면 0
        my_status = 1.0 if self.env.read_m(0xCB1A) > 0 else 0.0
        en_status = 1.0 if self.env.read_m(0xD0FD) > 0 else 0.0

        # 6. 최종 11차원 벡터 생성
        # [내HP%, 상대HP%, 레벨비중, 내상태, 상대상태, 기술1유무, 기술2유무, 기술3유무, 기술4유무, 뱃지수, 이벤트수]
        badges = self.env.get_badges() / 8.0
        events = self.env.get_all_events_reward() / 100.0 # 대략적인 스케일링
        
        vector = [
            my_hp_pct, en_hp_pct, lvl_ratio, 
            my_status, en_status,
            has_moves[0], has_moves[1], has_moves[2], has_moves[3],
            badges, events
        ]
        
        return np.array(vector, dtype=np.float32)