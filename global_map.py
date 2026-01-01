import os
import json

MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")
PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD
MAP_CENTER = (GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2)

# map_data.json 로드
try:
    with open(MAP_PATH, "r", encoding="utf-8") as map_data:
        map_json = json.load(map_data)
        # regions 키가 있으면 그것을 쓰고, 없으면 파일 전체를 리스트로 가정
        if "regions" in map_json:
            MAP_DATA = map_json["regions"]
        else:
            MAP_DATA = map_json
            
    # 리스트를 딕셔너리로 변환 (ID를 키로 사용)
    MAP_DATA = {int(e["id"]): e for e in MAP_DATA}
    
except Exception as e:
    print(f"Warning: map_data.json load failed or empty. ({e})")
    MAP_DATA = {}

def get_map_id_from_mem(map_group, map_number):
    """
    게임 메모리의 Map Group과 Map Number를 결합하여 고유 ID를 반환
    """
    return (map_group << 8) | map_number

def local_to_global(r: int, c: int, map_n: int):
    """
    로컬 좌표를 글로벌 좌표로 변환.
    데이터에 없는 맵(실내 등)이 들어오면 에러를 내지 않고 중앙 좌표를 반환.
    """
    try:
        # 1. 맵 데이터에 없는 ID(6148 등 실내 맵)가 들어오면 방어 코드 작동
        if map_n not in MAP_DATA:
            # 너무 자주 뜨면 로그가 지저분하므로 필요할 때만 주석 해제하세요
            # print(f"Unknown Map ID: {map_n} (Usually Indoor) -> Returning Center")
            return MAP_CENTER
            
        # 2. 데이터가 있으면 정상 변환
        map_x, map_y = MAP_DATA[map_n]["coordinates"]
        
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        
        # 3. 글로벌 맵 범위를 벗어나는지 체크
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
            
        return MAP_CENTER
        
    except Exception:
        # 그 외 모든 에러 상황에서도 멈추지 않고 중앙 좌표 반환
        return MAP_CENTER