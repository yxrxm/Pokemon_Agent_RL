# # metamon_runtime/bc/dataset.py
# import torch
# from torch.utils.data import Dataset

# class ScreenActionDataset(Dataset):
#     """
#     (screen, action) 데이터셋
#     action:
#       0 = A
#       1 = UP
#       2 = DOWN
#     """

#     def __init__(self, screens, actions):
#         assert len(screens) == len(actions)
#         self.screens = screens
#         self.actions = actions

#     def __len__(self):
#         return len(self.screens)

#     def __getitem__(self, idx):
#         # screen: (H, W, C) -> (C, H, W)
#         screen = self.screens[idx].permute(2, 0, 1) / 255.0
#         action = self.actions[idx]
#         return screen.float(), torch.tensor(action, dtype=torch.long)

import os
import json
import lz4.frame
import torch
from tqdm import tqdm

from metamon.metamon.interface import UniversalState
from encoders import encode_state
from action_codec import encode_action

DATA_ROOT = "../data/gen2ou"
OUT_PATH = "./data/bc_dataset.pt"

os.makedirs("./data", exist_ok=True)

states_out = []
actions_out = []

files = [f for f in os.listdir(DATA_ROOT) if f.endswith(".json.lz4")]

print(f"총 {len(files)}개 리플레이 처리 시작")

for fname in tqdm(files):
    path = os.path.join(DATA_ROOT, fname)

    with lz4.frame.open(path, "rb") as f:
        raw = json.loads(f.read().decode("utf-8"))

    states = [UniversalState.from_dict(s) for s in raw["states"]]
    actions = raw["actions"]

    for s, a in zip(states, actions):
        act = encode_action(a)
        if act is None:
            continue

        vec = encode_state(s)
        states_out.append(vec)
        actions_out.append(act)

print(f"유효 샘플 수: {len(states_out)}")

torch.save(
    {
        "states": torch.stack(states_out),
        "actions": torch.tensor(actions_out, dtype=torch.long),
    },
    OUT_PATH
)

print(f"BC 데이터셋 저장 완료 → {OUT_PATH}")
