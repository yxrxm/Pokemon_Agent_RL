# # metamon_runtime/bc/model.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BCPolicyCNN(nn.Module):
#     """
#     화면 → 행동(A / UP / DOWN) 분류 모델
#     """

#     def __init__(self):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU(),
#         )

#         # 입력 크기 자동 계산
#         with torch.no_grad():
#             dummy = torch.zeros(1, 3, 144, 160)  # GameBoy 화면 크기
#             n_flat = self.conv(dummy).view(1, -1).size(1)

#         self.fc = nn.Sequential(
#             nn.Linear(n_flat, 256),
#             nn.ReLU(),
#             nn.Linear(256, 3)  # A / UP / DOWN
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
#         return self.fc(x)

import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)
