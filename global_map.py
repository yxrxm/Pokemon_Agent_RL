import numpy as np

# 맵 파일이 없으므로 임의의 충분히 큰 크기로 설정
GLOBAL_MAP_SHAPE = (800, 800)

def get_map_id_from_mem(map_group, map_number):
    """
    단순히 (그룹 * 100 + 번호) 공식을 사용하여 고유성을 보장합니다.
    예: 그룹 3, 번호 5 -> ID 305
    """
    return int(map_group) * 100 + int(map_number)

def local_to_global(y, x, map_id):
    # 맵의 중앙 좌표
    center_y = GLOBAL_MAP_SHAPE[0] // 2
    center_x = GLOBAL_MAP_SHAPE[1] // 2
    
    # 오프셋 없이 단순히 중앙 + 로컬좌표 반환
    return center_y + y, center_x + x