import numpy as np
import mujoco
from gymnasium import utils as gym_utils
from gymnasium.envs.mujoco import MujocoEnv
from utils import quat2expmap

class ModularEnv(MujocoEnv, gym_utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml,  **kwargs):
        self.xml = xml
        super().__init__(model_path=xml, frame_skip=4, observation_space=None,  **kwargs)
        gym_utils.EzPickle.__init__(self, xml)

    def step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.data.qpos[0]
        torso_height, torso_ang = self.data.qpos[1:3]
        reward = (posafter - posbefore) / self.dt
        reward += 1.0
        reward -= 1e-3 * float(np.square(a).sum())
        terminated = not (torso_height > (0.8 - 0.26) and torso_height < (2.0 - 0.26)
                          and torso_ang > -1.0 and torso_ang < 1.0)
        truncated = False
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}

    # helpers
    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _body_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)]

    def _get_obs(self):
        def limb_type_vec(b):
            if b == 'torso': return np.array((1,0,0,0), np.float32)
            if '1' in b:     return np.array((0,1,0,0), np.float32)
            if '2' in b:     return np.array((0,0,1,0), np.float32)
            if '3' in b:     return np.array((0,0,0,1), np.float32)
            return np.array((0,0,0,0), np.float32)

        torso_id = self._body_id('torso')
        torso_x  = self.data.xpos[torso_id, 0]

        def obs_per_limb(b):
            bid  = self._body_id(b)
            xpos = self.data.xpos[bid].copy(); xpos[0] -= torso_x
            q    = self.data.xquat[bid].copy()
            v6   = self.data.cvel[bid].copy()
            vang = v6[:3]; vlin = np.clip(v6[3:], -10, 10)
            expmap = quat2expmap(q)
            ltype  = limb_type_vec(b)

            if b == 'torso':
                angle = 0.0; jrange = np.array([0.0, 0.0], np.float32)
            else:
                jadr = self.model.body_jntadr[bid]
                qadr = self.model.jnt_qposadr[jadr]
                angle  = np.degrees(self.data.qpos[qadr])
                jrange = np.degrees(self.model.jnt_range[jadr]).astype(np.float32)
                angle  = (angle - jrange[0]) / (jrange[1] - jrange[0] + 1e-8)
                jrange = (np.array([180.0,180.0], np.float32) + jrange) / 360.0

            return np.concatenate([xpos, vlin, vang, expmap, ltype, [angle], jrange]).astype(np.float32)

        bodies = self._body_names()[1:]
        return np.concatenate([obs_per_limb(b) for b in bodies]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        try:
            viewer = self.mujoco_renderer.viewer
            if viewer is not None:
                viewer.cam.trackbodyid = 2
                viewer.cam.distance = self.model.stat.extent * 0.5
                viewer.cam.lookat[2] = 1.15
                viewer.cam.elevation = -20
        except Exception:
            pass
