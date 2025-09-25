# policies/sac.py
from __future__ import annotations
import os
import numpy as np
from typing import Dict

from stable_baselines3 import SAC as SB3_SAC


class SACPolicy:
    """
    Stable-Baselines3 SAC inference 래퍼.
    - collector는 act(obs_valid)만 호출하면 됨.
    - obs_valid shape = (L * limb_obs_size,)
    - 반환 action shape = (L,), 값 범위 [-1, 1]

    사용 예:
        model_paths = {4: "ckpt_L4.zip", 6: "ckpt_L6.zip"}
        policy = SACPolicy(limb_obs_size=19, model_paths_by_L=model_paths, deterministic=False)
        a = policy.act(obs_valid)  # (-1,1)^L
    """

    def __init__(
        self,
        limb_obs_size: int,
        model_paths_by_L: Dict[int, str],
        deterministic: bool = False,
    ) -> None:
        self.limb_obs_size = int(limb_obs_size)
        self.deterministic = bool(deterministic)
        self.models: Dict[int, SB3_SAC] = {}

        if not model_paths_by_L:
            raise ValueError("model_paths_by_L를 제공해야 합니다. 예: {4:'ckpt_L4.zip', 6:'ckpt_L6.zip'}")

        # L별로 SB3 모델 로드
        for L, path in model_paths_by_L.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[SACPolicy] 모델 파일을 찾을 수 없습니다: L={L}, path={path}")
            self.models[L] = SB3_SAC.load(path, print_system_info=False)

    def _infer_L(self, obs_valid: np.ndarray) -> int:
        n = int(obs_valid.size)
        if n % self.limb_obs_size != 0:
            raise ValueError(
                f"[SACPolicy] obs_valid 길이({n})가 limb_obs_size({self.limb_obs_size})로 나누어떨어지지 않습니다."
            )
        return n // self.limb_obs_size

    def act(self, obs_valid: np.ndarray) -> np.ndarray:
        """
        obs_valid: (L * limb_obs_size,)
        return: (L,), in [-1, 1]
        """
        if obs_valid.ndim != 1:
            obs_valid = np.asarray(obs_valid).reshape(-1)

        L = self._infer_L(obs_valid)
        model = self.models.get(L, None)
        
        # SB3 predict는 (batch, obs_dim) 입력을 기대
        action, _ = model.predict(obs_valid[None, :], deterministic=self.deterministic)
        action = np.asarray(action).reshape(-1)

        # collector 계약: [-1, 1]^L 반환
        if action.size != L:
            raise RuntimeError(f"[SACPolicy] 예측 action 크기({action.size})가 L({L})과 다릅니다.")
        return np.clip(action.astype(np.float32), -1.0, 1.0)
