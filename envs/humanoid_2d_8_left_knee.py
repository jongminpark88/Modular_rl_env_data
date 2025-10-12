import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium import utils as gym_utils
from gymnasium.envs.mujoco import MujocoEnv
from utils import quat2expmap


class ModularEnv(MujocoEnv, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    FRAME_SKIP = 4
    ALIVE_BONUS = 1.0
    CTRL_COST_COEF = 1e-3
    HEIGHT_MIN, HEIGHT_MAX = 0.4, 2.1
    PITCH_ABS_MAX = 1.0  # rad

    def __init__(self, xml, **kwargs):
        self.xml = xml
        super().__init__(model_path=xml, frame_skip=self.FRAME_SKIP,
                         observation_space=None, **kwargs)
        gym_utils.EzPickle.__init__(self, xml)

        mujoco.mj_forward(self.model, self.data)
        obs0 = np.asarray(self._get_obs(), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32
        )

    # -------- helpers --------
    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _body_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)]
    # -------------------------

    def _get_obs(self):
        def limb_type_vec(b: str):
            if 'hip' in b:
                return np.array((1, 0, 0, 0), np.float32)
            elif 'knee' in b:
                return np.array((0, 1, 0, 0), np.float32)
            elif 'shoulder' in b:
                return np.array((0, 0, 1, 0), np.float32)
            elif 'elbow' in b:
                return np.array((0, 0, 0, 1), np.float32)
            else:
                return np.array((0, 0, 0, 0), np.float32)

        torso_x = self.data.xpos[self._body_id('torso'), 0]

        def obs_per_limb(b: str):
            bid = self._body_id(b)

            # 위치: torso x 기준 상대
            xpos = self.data.xpos[bid].copy()
            xpos[0] -= torso_x

            # 속도: 바디 프레임(cvel) — 치타/워커와 동일
            v6 = self.data.cvel[bid].copy()  # [ang(3), lin(3)] in body frame
            vang = v6[:3]
            vlin = np.clip(v6[3:], -10, 10)

            # 회전 → expmap
            q = self.data.xquat[bid].copy()
            expmap = quat2expmap(q)

            ltype = limb_type_vec(b)

            # 각/범위: 바디 첫 조인트만 사용(없으면 0)
            if b == 'torso':
                angle = 0.0
                jrange = np.array([0.0, 0.0], np.float32)
            else:
                jadr = int(self.model.body_jntadr[bid])
                if jadr == -1 or jadr >= self.model.njnt:
                    angle = 0.0
                    jrange = np.array([0.0, 0.0], np.float32)
                else:
                    qadr = int(self.model.jnt_qposadr[jadr])  # 1-DoF 가정
                    angle = float(np.degrees(self.data.qpos[qadr]))
                    jrange = np.degrees(self.model.jnt_range[jadr]).astype(np.float32)
                    # 정규화: (ε 포함) — 다른 환경들과 완전 동일
                    angle = (angle - jrange[0]) / (jrange[1] - jrange[0] + 1e-8)
                    jrange = (np.array([180.0, 180.0], np.float32) + jrange) / 360.0

            return np.concatenate([xpos, vlin, vang, expmap, ltype, [angle], jrange]).astype(np.float32)

        bodies = self._body_names()[1:]  # world 제외
        full = np.concatenate([obs_per_limb(b) for b in bodies])
        return full.ravel()

    def step(self, action):
        pos_before = float(self.data.qpos[0])
        self.do_simulation(action, self.frame_skip)
        pos_after = float(self.data.qpos[0])
        height, ang = float(self.data.qpos[1]), float(self.data.qpos[2])

        reward = (pos_after - pos_before) / self.dt
        reward += self.ALIVE_BONUS
        reward -= self.CTRL_COST_COEF * float(np.square(action).sum())

        alive = (self.HEIGHT_MIN < height < self.HEIGHT_MAX) and (abs(ang) < self.PITCH_ABS_MAX)
        terminated = not alive
        truncated = False

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        # Gymnasium에서는 viewer가 없을 수 있으므로 안전 처리(다른 환경과 동일)
        try:
            viewer = self.mujoco_renderer.viewer
            if viewer is not None:
                viewer.cam.trackbodyid = 1
                viewer.cam.distance = self.model.stat.extent * 1.0
                viewer.cam.lookat[2] = 2.0
                viewer.cam.elevation = -20
        except Exception:
            pass
