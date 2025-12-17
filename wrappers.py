# wrappers.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
import utils


class ModularEnvWrapper(gym.Wrapper):
    """
    - obs를 고정 길이(obs_max_len)로 패딩
    - 모듈러 정책 액션 순서를 MuJoCo env의 액션 순서(motors)에 매핑
    - Gymnasium 규약(step: terminated/truncated, reset: (obs, info))
    """

    def __init__(self, env, obs_max_len=None):
        super().__init__(env)
        # 초기 reset 한 번 (Gymnasium)
        obs, info = self.env.reset()
        base_obs_len = obs.shape[0]

        self.obs_max_len = int(obs_max_len) if obs_max_len else int(base_obs_len)
        self.action_len = self.env.action_space.shape[0]
        self.num_limbs = len(self.env.model.body_names[1:])
        self.limb_obs_size = base_obs_len // self.num_limbs
        self.max_action = float(self.env.action_space.high[0])
        self.xml = getattr(self.env, "xml", None)

        # 모터 순서 매핑
        self.motors = utils.getMotorJoints(self.env.xml)
        self.joints = utils.getGraphJoints(self.env.xml)
        self.action_order = [-1] * self.num_limbs
        for i in range(len(self.joints)):
            assert sum([j in self.motors for j in self.joints[i][1:]]) <= 1, (
                "Modular policy does not support two motors per body"
            )
            for j in self.joints[i]:
                if j in self.motors:
                    self.action_order[i] = self.motors.index(j)
                    break

    def step(self, action):
        # 1) 벡터 env 패딩 제거
        action = np.asarray(action[: self.num_limbs], dtype=np.float32)
        # 2) env의 motor 순서로 매핑 (기본 0.0)
        env_action = np.zeros(len(self.motors), dtype=np.float32)
        for i in range(self.num_limbs):
            idx = self.action_order[i]
            if idx >= 0:
                env_action[idx] = action[i]

        obs, reward, terminated, truncated, info = self.env.step(env_action)
        done = bool(terminated or truncated)

        assert len(obs) <= self.obs_max_len, (
            f"env obs length {len(obs)} exceeds obs_max_len {self.obs_max_len}"
        )
        if len(obs) < self.obs_max_len:
            pad = np.zeros(self.obs_max_len - len(obs), dtype=obs.dtype)
            obs = np.concatenate([obs, pad], axis=0)
        else:
            obs = obs[: self.obs_max_len]

        # Gym-style로 넘기고 싶은데, 수집기는 done을 기대하므로 info에 포함해도 되지만
        # 여기선 관례 유지 차원에서 Gym 스타일 반환으로 맞춘다:
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        assert len(obs) <= self.obs_max_len, (
            f"env obs length {len(obs)} exceeds obs_max_len {self.obs_max_len}"
        )
        if len(obs) < self.obs_max_len:
            pad = np.zeros(self.obs_max_len - len(obs), dtype=obs.dtype)
            obs = np.concatenate([obs, pad], axis=0)
        else:
            obs = obs[: self.obs_max_len]
        # 수집기는 (obs)만 기대하므로 obs만 반환
        return obs


class IdentityWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs, info = self.env.reset()
        self.num_limbs = len(self.env.model.body_names[1:])
        self.limb_obs_size = obs.shape[0] // self.num_limbs
        self.max_action = float(self.env.action_space.high[0])


class ResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def frame(self, qpos, qvel):
        self.set_state(qpos, qvel)
