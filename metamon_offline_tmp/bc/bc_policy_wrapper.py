import torch
from .bc_model import BCPolicy


class BCPolicyWrapper:
    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model = BCPolicy(
            obs_dim=ckpt["obs_dim"],
            n_actions=4
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def predict(self, obs_vec):
        """
        obs_vec: torch.Tensor, shape [obs_dim]
        return: int (0~3)
        """
        with torch.no_grad():
            logits = self.model(obs_vec.unsqueeze(0))
            return logits.argmax(dim=1).item()
