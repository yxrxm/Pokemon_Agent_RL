import os
import re

# ==========================================
# [메모리 주소 모음] (Memory Addresses)
# 포켓몬 골드/실버 버전 (영문판) 기준
# ==========================================

# 1. 플레이어 상태
MEM_MONEY = 0xD84E  # 돈 (3 bytes, BCD)
MEM_BADGES = 0xD857  # 획득한 배지 플래그 (Johto)
MEM_ID = 0xD1A3  # 트레이너 ID (2 bytes)
MEM_NAME = 0xD158  # 트레이너 이름

# 2. 위치 정보
MEM_MAP_GROUP = 0xDCB5  # 현재 맵 그룹
MEM_MAP_NUMBER = 0xDCB6  # 현재 맵 번호
MEM_Y_POS = 0xDCB7  # Y 좌표
MEM_X_POS = 0xDCB8  # X 좌표

# 3. 파티(포켓몬) 정보
MEM_PARTY_COUNT = 0xDCD7  # 현재 데리고 있는 포켓몬 수
MEM_PARTY_LEVELS = 0xDCDF  # 파티 포켓몬들의 레벨 시작 지점 (순서대로 1바이트씩)

# 4. 전투 관련
MEM_BATTLE_TYPE = 0xD22D  # 0이면 필드, 0이 아니면 전투/메뉴 등


# ==========================================
# [도구 함수] (Helper Functions)
# PyBoy 메모리 읽기 및 변환
# ==========================================

def read_uint8(pyboy, address):
    """
    1바이트(8비트) 정수를 읽어옵니다.
    """
    return pyboy.memory[address]


def read_uint16(pyboy, address):
    """
    2바이트(16비트) 정수를 읽어옵니다. (리틀 엔디안)
    """
    low = pyboy.memory[address]
    high = pyboy.memory[address + 1]
    return (high << 8) + low


def read_bcd(pyboy, address, length):
    """
    BCD(Binary Coded Decimal) 형식의 값을 읽어 일반 숫자로 변환합니다.
    """
    value = 0
    for i in range(length):
        byte = pyboy.memory[address + i]
        value = (value * 100) + ((byte >> 4) * 10) + (byte & 0x0F)
    return value


def count_set_bits(value):
    """
    이진수에서 1의 개수를 셉니다. (배지 개수 파악용)
    """
    return bin(value).count('1')


def get_badges(pyboy):
    """
    현재 획득한 배지(성도 지방) 개수를 반환합니다.
    """
    return count_set_bits(read_uint8(pyboy, MEM_BADGES))


def get_level_sum(pyboy):
    """
    현재 데리고 있는 모든 포켓몬의 레벨 합계를 구합니다.
    (게임 진행도 파악용 지표)
    """
    try:
        party_count = read_uint8(pyboy, MEM_PARTY_COUNT)
        level_sum = 0
        # 파티 포켓몬 수만큼 루프를 돌며 레벨을 더함
        for i in range(party_count):
            level = read_uint8(pyboy, MEM_PARTY_LEVELS + i)
            level_sum += level
        return level_sum
    except Exception:
        return 0


# ==========================================
# [파일 관리 헬퍼 함수]
# 모델 및 세션 번호 자동 관리
# ==========================================

def get_next_index(directory, prefix):
    """
    폴더 안에서 prefix_1, prefix_2 ... 패턴을 찾아 다음 번호를 리턴합니다.
    (세션 폴더 생성 시 사용)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 1

    max_idx = 0
    pattern = re.compile(rf"^{prefix}_(\d+)")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx

    return max_idx + 1


def get_latest_model_path(directory, prefix):
    """
    폴더에서 가장 높은 번호의 일반 모델 경로를 찾습니다.
    (final_model_1.zip, final_model_2.zip ...)
    """
    if not os.path.exists(directory):
        return None, 0

    max_idx = 0
    latest_file = None
    pattern = re.compile(rf"^{prefix}_(\d+)")

    for filename in os.listdir(directory):
        if filename.endswith(".zip"):
            match = pattern.match(filename)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
                    latest_file = os.path.join(directory, filename)

    return latest_file, max_idx


def get_best_badge_model(directory):
    """
    ★ [커리큘럼 학습용]
    models 폴더에서 'final_model_badge_X.zip' 중 X(배지 수)가 가장 큰 파일을 찾습니다.
    리턴값: (파일경로, 배지개수)
    """
    if not os.path.exists(directory):
        return None, 0

    max_badge = -1
    best_model_path = None

    # 정규표현식: final_model_badge_숫자.zip
    pattern = re.compile(r"^final_model_badge_(\d+)\.zip$")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            badge_num = int(match.group(1))
            # 배지 개수가 더 많으면 갱신
            if badge_num > max_badge:
                max_badge = badge_num
                best_model_path = os.path.join(directory, filename)

    return best_model_path, max_badge