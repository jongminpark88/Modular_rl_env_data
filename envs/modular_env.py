import os
import re
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# ---------- utils: identical to repo semantics ----------
def quat2expmap(q):
    """
    레포의 quat2expmap과 같은 수학적 동작을 하도록 구현
    (정규화 허용오차/폴딩 로직 등 동일)
    """
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if np.abs(n - 1.0) > 1e-3:
        if n > 0:
            q = q / n
        else:
            return np.zeros(3, dtype=np.float32)

    w, x, y, z = q
    sin_half = np.linalg.norm([x, y, z])
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)

    axis = np.array([x, y, z], dtype=np.float64) / sin_half
    theta = 2.0 * np.arctan2(sin_half, w)
    theta = np.mod(theta + 2*np.pi, 2*np.pi)
    if theta > np.pi:
        theta = 2*np.pi - theta
        axis = -axis
    r = axis * theta
    return r.astype(np.float32)

def limb_one_hot(name: str):
    n = name.lower()
    if "torso" in n or n == "torso":
        return np.array((1,0,0,0), dtype=np.float32)
    if re.search(r"thigh|hip", n):
        return np.array((0,1,0,0), dtype=np.float32)
    if re.search(r"shin|knee|leg", n):
        return np.array((0,0,1,0), dtype=np.float32)
    if re.search(r"foot|ankle", n):
        return np.array((0,0,0,1), dtype=np.float32)
    return np.zeros(4, dtype=np.float32)

# ---------- Env compatible with modular-rl obs/action ----------
class ModularEnv(gym.Env):
    """
    modular-rl 레포와 obs, action 스펙을 동일하게 맞춘 버전.
      - action: policy 출력 a를 그대로 사용(스케일링 없음) -> XML motor ctrlrange가 [-1,1]이어야 함.
      - obs: per-limb 특징(상대 위치, xvelp/xvelr, expmap, (각/범위 정규화), limb type one-hot)
      - reward: forward/dt + alive_bonus - 1e-3 * sum(a^2)
      - done: 항상 False (TimeLimit 래퍼 사용)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, xml_path: str, frame_skip: int = 4, render_mode=None,
                 vel_clip: float = 10.0, alive_bonus: float = 1.0):
        super().__init__()
        
        self.xml_path = xml_path
        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self.vel_clip = float(vel_clip)
        self.alive_bonus = float(alive_bonus)

        # MuJoCo setup
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # action: 레포와 동일하게 [-1,1]^nu를 그대로 ctrl에 넣는다(스케일링 없음).
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # obs: per-limb 길이를 동적으로 계산 (한 번 forward하여 차원 산정)
        mujoco.mj_forward(self.model, self.data)
        sample = self._build_obs()  # 레포와 동일 로직
        self._obs_dim = int(sample.size)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        self._renderer = None
        self._x_before = None  # forward 보상 계산을 위해 저장

    # ------- per-limb obs identical to repo -------
    def _build_obs(self):
        # torso world x
        # (repo는 torso body 이름을 'torso'로 가정. 없으면 두번째 바디(id=1)를 torso로 본다)
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if torso_id < 0:
            torso_id = 1
        torso_x = float(self.data.xpos[torso_id, 0])

        feats = []
        # world body(0) 제외, 1..nbody-1 순서 == repo의 self.model.body_names[1:]
        for b_id in range(1, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b_id) or f"body{b_id}"

            # xpos (x만 torso 기준으로 이동)
            xpos = self.data.xpos[b_id].copy()
            xpos[0] -= torso_x

            # 선속도(clip) / 각속도
            # repo는 data.get_body_xvelp/r(b) 사용 → mujoco 2.3에선 xvelp/xvelr에 해당
            vlin = np.clip(self.data.xvelp[b_id], -self.vel_clip, self.vel_clip).astype(np.float32)
            vang = self.data.xvelr[b_id].astype(np.float32)

            # expmap
            expm = quat2expmap(self.data.xquat[b_id])

            # joint angle / range 정규화 (바디당 1 조인트 가정, torso는 0)
            if b_id == torso_id:
                angle = np.array([0.0], dtype=np.float32)
                jrng_n = np.array([0.0, 0.0], dtype=np.float32)
            else:
                jadr = self.model.body_jntadr[b_id]
                jnum = self.model.body_jntnum[b_id]
                if jnum >= 1:
                    j_id = jadr  # 첫 번째 조인트
                    qadr = self.model.jnt_qposadr[j_id]
                    jrng = self.model.jnt_range[j_id].copy()
                    # 도 단위로 맞춘 다음 레포와 동일한 정규화
                    ang_deg = np.degrees(self.data.qpos[qadr])
                    rng_deg = np.degrees(jrng)
                    denom = (rng_deg[1] - rng_deg[0]) if (rng_deg[1] - rng_deg[0]) != 0 else 1.0
                    angle = np.array([(ang_deg - rng_deg[0]) / denom], dtype=np.float32)
                    jrng_n = np.array([(180. + rng_deg[0]) / 360., (180. + rng_deg[1]) / 360.], dtype=np.float32)
                else:
                    angle = np.array([0.0], dtype=np.float32)
                    jrng_n = np.array([0.0, 0.0], dtype=np.float32)

            # limb type one-hot (repo SMP 코드 버전과 일치)
            limb_oh = limb_one_hot(name)

            obs_limb = np.concatenate([xpos.astype(np.float32),
                                       vlin, vang,
                                       expm,
                                       angle, jrng_n,
                                       limb_oh], dtype=np.float32)
            feats.append(obs_limb)

        full = np.concatenate(feats, axis=0).astype(np.float32)
        return full

    def _get_obs(self):
        return self._build_obs()

    # ------- Gymnasium step/reset -------
    def step(self, action):
        # 레포와 동일: a를 그대로 ctrl로 사용 (스케일링 없음)
        a = np.asarray(action, dtype=np.float32)
        self.data.ctrl[:] = a

        # before
        if self._x_before is None:
            self._x_before = float(self.data.qpos[0])
        before = self._x_before

        # simulate
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # after
        after = float(self.data.qpos[0])
        self._x_before = after

        # dt (레포는 env.dt 사용; 여기선 timestep*frame_skip)
        dt = self.model.opt.timestep * self.frame_skip
        forward = (after - before) / max(dt, 1e-8)

        # reward: forward + alive - 1e-3 * ||a||^2  (레포식)
        ctrl_cost = 1e-3 * float(np.square(a).sum())
        reward = forward + self.alive_bonus - ctrl_cost

        obs = self._get_obs()
        terminated = False   # 레포 동일
        truncated = False    # TimeLimit 래퍼 사용 권장
        info = {"reward_run": forward, "reward_ctrl": -ctrl_cost}

        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 레포는 init_qpos/vel 주변 작은 노이즈. keyframe 없으면 0으로.
        if self.model.nkey > 0:
            qpos0 = self.model.key_qpos.copy()
            qvel0 = self.model.key_qvel.copy()
        else:
            qpos0 = np.zeros(self.model.nq, dtype=np.float64)
            qvel0 = np.zeros(self.model.nv, dtype=np.float64)

        self.data.qpos[:] = qpos0 + self.np_random.uniform(low=-5e-3, high=5e-3, size=self.model.nq)
        self.data.qvel[:] = qvel0 + self.np_random.uniform(low=-5e-3, high=5e-3, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self._x_before = float(self.data.qpos[0])

        return self._get_obs(), {}
    def render(self):
        if self.render_mode == "human":
            # 수동 뷰어(별도 창) — 서버 환경이면 사용 불가할 수 있음
            try:
                if self._renderer is None:
                    self._renderer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception:
                pass
            return None
        elif self.render_mode == "rgb_array":
            try:
                if self._renderer is None:
                    from mujoco import mj_renderer
                    self._renderer = mj_renderer.Renderer(self.model)
                self._renderer.update_scene(self.data)
                img = self._renderer.render()
                return img
            except Exception:
                return None
        else:
            return None

    def close(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
