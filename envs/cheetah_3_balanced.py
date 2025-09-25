import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium import utils as gym_utils
from gymnasium.envs.mujoco import MujocoEnv
from utils import quat2expmap


class ModularEnv(MujocoEnv, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml,  **kwargs):
        self.xml = xml
        super().__init__(model_path=xml, frame_skip=4, observation_space=None,  **kwargs)
        gym_utils.EzPickle.__init__(self, xml)

        mujoco.mj_forward(self.model, self.data)
        
        obs0 = self._get_obs()
        obs0 = np.asarray(obs0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs0.shape, dtype=np.float32)


    def step(self, action):
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        reward_ctrl = -0.1 * float(np.square(action).sum())
        reward_run  = float((xposafter - xposbefore) / self.dt)
        reward = reward_ctrl + reward_run

        obs = self._get_obs()
        terminated = False
        truncated  = False
        info = {"reward_run": reward_run, "reward_ctrl": reward_ctrl}
        return obs, reward, terminated, truncated, info

    # --- dm-mujoco helpers ---
    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _body_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)]
    # --------------------------

    def _get_obs(self):
        def limb_type_vec(bname: str):
            if bname == "torso": return np.array((1, 0, 0, 0), dtype=np.float32)
            if "thigh" in bname: return np.array((0, 1, 0, 0), dtype=np.float32)
            if "shin"  in bname: return np.array((0, 0, 1, 0), dtype=np.float32)
            if "foot"  in bname: return np.array((0, 0, 0, 1), dtype=np.float32)
            return np.array((0, 0, 0, 0), dtype=np.float32)

        torso_id = self._body_id("torso")
        torso_x  = self.data.xpos[torso_id, 0]

        def obs_per_limb(b):
            bid  = self._body_id(b)
            xpos = self.data.xpos[bid].copy()
            xpos[0] -= torso_x
            q   = self.data.xquat[bid].copy()
            v6  = self.data.cvel[bid].copy()   # [ang(3), lin(3)]
            vang = v6[:3]
            vlin = np.clip(v6[3:], -10, 10)
            expmap = quat2expmap(q)
            ltype  = limb_type_vec(b)

            if b == "torso":
                angle, jrange = 0.0, np.array([0.0, 0.0], dtype=np.float32)
            else:
                jadr = self.model.body_jntadr[bid]
                qadr = self.model.jnt_qposadr[jadr]
                angle  = np.degrees(self.data.qpos[qadr])
                jrange = np.degrees(self.model.jnt_range[jadr]).astype(np.float32)
                angle  = (angle - jrange[0]) / (jrange[1] - jrange[0] + 1e-8)
                jrange = (np.array([180.0, 180.0], np.float32) + jrange) / 360.0

            return np.concatenate([xpos, vlin, vang, expmap, ltype, [angle], jrange]).astype(np.float32)

        bodies = self._body_names()[1:]  # skip world
        full = np.concatenate([obs_per_limb(b) for b in bodies])
        return full.ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
