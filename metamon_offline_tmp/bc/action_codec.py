# def encode_action(action):
#     """
#     metamon action → 정수 라벨
#     MOVE_1~MOVE_4만 사용 (교체는 일단 버림)
#     """

#     if action["type"] == "move":
#         idx = action.get("move_index", None)
#         if idx is not None and 0 <= idx <= 3:
#             return idx  # 0~3

#     return None  # 학습에서 제외

def encode_action(action):
    """
    metamon action → 정수 라벨
    action이 int면 그대로 사용
    """

    # 경우 1: 이미 int인 경우 (가장 흔함)
    if isinstance(action, int):
        if 0 <= action <= 3:
            return action
        else:
            return None

    # 경우 2: dict 형태인 경우 (일부 리플레이)
    if isinstance(action, dict):
        if action.get("type") == "move":
            idx = action.get("move_index", None)
            if idx is not None and 0 <= idx <= 3:
                return idx

    return None
