import os
import re

# [메모리 주소 모음] (Memory Addresses)
# 포켓몬 골드/실버 버전 (영문판) 기준

# 1. 플레이어 상태
MEM_MONEY = 0xD573  # 돈 (BCD 포맷)
MEM_BADGES = 0xD57C  # 획득한 배지 플래그 (Johto)
MEM_BATTLE_TYPE = 0xD116  # 0:필드, 1:야생전투, 2:트레이너전투

# 2. 위치 정보
MEM_MAP_GROUP = 0xDA00  # 현재 맵 그룹
MEM_MAP_NUMBER = 0xDA01  # 현재 맵 번호
MEM_X_POS = 0xD20D  # Y 좌표
MEM_Y_POS = 0xD20E  # X 좌표

# --- [C] 필드/파티 정보 (Field / Party Slot 1) ---
# 전투가 아닐 때(필드) 관리하는 메인 포켓몬 정보
PARTY_STRUCT_SIZE = 48
# (독 데미지, 회복 감지용)
MEM_PARTY_COUNT    = 0xDA22  # 파티에 있는 포켓몬 수
MEM_PARTY_LEVELS = 0xDA49    # 파티 레벨 시작 주소 1번과 동일
MEM_P1_LEVEL       = 0xDA49  # 1번 포켓몬 레벨
MEM_P1_HP          = 0xDA4C  # 1번 포켓몬 현재 체력 (Little Endian)
MEM_P1_MAX_HP      = 0xDA4E  # 1번 포켓몬 최대 체력 (Little Endian)
MEM_P1_EXP         = 0xDA32  # 1번 포켓몬 경험치 (3 Bytes, Big Endian)
OFFSET_LEVEL = 0x1F

# --- [D] 전투 전용 정보 (Active Battle Pokemon) ---
# 전투 중에만 유효함. 교체를 해도 '현재 나와있는 놈'의 정보가 됨.
# 주의: 이 영역의 HP는 Big Endian일 가능성이 높음.
MEM_BATTLE_LEVEL   = 0xCB19
MEM_BATTLE_HP_NOW  = 0xCB1C  # 현재 싸우는 포켓몬 HP (Big Endian 추정)
MEM_BATTLE_HP_MAX  = 0xCB1E  # 현재 싸우는 포켓몬 Max HP

# --- [E] 적 포켓몬 정보 (Enemy Battle Pokemon) ---
MEM_ENEMY_LEVEL    = 0xD0FC
MEM_ENEMY_HP       = 0xD0FF  # 적 HP (Big Endian 추정)
MEM_ENEMY_MAX_HP   = 0xD101  # 적 Max HP


# [도구 함수] (Helper Functions)
# PyBoy 메모리 읽기 및 변환

def read_uint8(pyboy, address):
    """
    1바이트(8비트) 정수를 읽어옵니다.
    """
    return pyboy.memory[address]


#HP, 좌표 이런 것들은 덧셈/뺄셈의 효율 때문에 리틀 엔디안을 사용함.
def read_uint16(pyboy, address):
    """
    2바이트(16비트) 정수를 읽어옵니다. (리틀 엔디안)
    좌표, HP 등등은 Little Endian
    """
    low = pyboy.memory[address]
    high = pyboy.memory[address + 1]
    return (high << 8) + low

def read_be16(pyboy, address):
    """
    2바이트(16비트) 정수를 읽어옵니다. (빅 엔디안 - Big Endian)
    전투 중 HP, 적 HP 등은 Big Endian을 사용합니다.
    """
    high = pyboy.memory[address]
    low = pyboy.memory[address + 1]
    return (high << 8) | low

#EXP, Money는 화면에 숫자를 보여주는 것이 중요해서 빅 엔디안을 사용함.
def read_uint24(pyboy, address):
    """3바이트 값 읽어서 정수로 변환 (빅 앤디안) // EXP, Money"""
    try:
        # Big Endian 방식 (앞주소가 큰 자릿수)
        h = pyboy.memory[address]  # 가장 높은 자릿수 (High)
        m = pyboy.memory[address + 1]  # 중간 자릿수 (Middle)
        l = pyboy.memory[address + 2]  # 낮은 자릿수 (Low)

        # 비트 시프트로 합치기
        return (h << 16) | (m << 8) | l
    except:
        return 0

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
        current_addr = MEM_PARTY_LEVELS

        for i in range(party_count):
            level = read_uint8(pyboy, current_addr)
            level_sum += level
            current_addr += PARTY_STRUCT_SIZE  # 다음 포켓몬 주소로 점프
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


def get_all_events_sum(pyboy):
    """
    0xD7B7 ~ 0xD8B6 구간의 모든 이벤트 플래그 비트 합(1의 개수)을 셉니다.
    스토리 진행도(체육관, 아이템 획득, NPC 대화 등)를 파악하는 핵심 지표입니다.
    """
    # 이벤트 플래그 메모리 범위
    START_ADDR = 0xD7B7
    END_ADDR = 0xD8B6

    total_events = 0

    # 범위 내의 모든 바이트를 순회
    for addr in range(START_ADDR, END_ADDR + 1):
        # 메모리 값 읽기 (0~255)
        val = pyboy.memory[addr]
        # 켜진 비트 수 세기 (예: 00000011 -> 2)
        total_events += bin(val).count('1')

    return total_events