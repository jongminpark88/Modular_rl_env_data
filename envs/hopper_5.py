import numpy as np
import mujoco                     
from gymnasium import utils as gym_utils
from gymnasium.envs.mujoco import MujocoEnv
from utils import quat2expmap


class ModularEnv(MujocoEnv, gym_utils.EzPickle):
    """
    종료 기준(원본 유지):
      height > 0.6, |ang| < 1.0, 상태 유한 & |s[2:]|<100
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml,  **kwargs):
        self.xml = xml
        super().__init__(model_path=xml, frame_skip=4, observation_space=None, **kwargs )
        gym_utils.EzPickle.__init__(self, xml)

    def step(self, a):
        posbefore = self.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter = self.data.qpos[0]

        torso_height, torso_ang = self.data.qpos[1:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * float(np.square(a).sum())

        s = np.concatenate([self.data.qpos.flat, self.data.qvel.flat])
        terminated = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (torso_height > 0.95)
            and (abs(torso_ang) < 1.0)
        )
        truncated = False
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), {}

    # ----- dm-mujoco 방식으로 치환한 부분 -----
    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _body_names(self):
        # 0: worldbody, 1부터 실제 본체들
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                for i in range(self.model.nbody)]

    def _get_obs(self):
        torso_id = self._body_id("torso")
        torso_x = self.data.xpos[torso_id, 0]

        def limb_type_vec(bname: str):
            if bname == "torso":     
                return np.array((1, 0, 0, 0), dtype=np.float32)
            if "thigh" in bname:     
                return np.array((0, 1, 0, 0), dtype=np.float32)
            if "leg"   in bname:     
                return np.array((0, 0, 1, 0), dtype=np.float32)
            if "foot"  in bname:     
                return np.array((0, 0, 0, 1), dtype=np.float32)
            return np.array((0, 0, 0, 0), dtype=np.float32)

        def obs_per_limb(bname: str):
            bid = self._body_id(bname)

            # 위치/자세/속도: dm-mujoco 필드 사용
            xpos  = self.data.xpos[bid].copy()
            xpos[0] -= torso_x
            q     = self.data.xquat[bid].copy()               # quaternion
            v6    = self.data.cvel[bid].copy()                # [ang(3), lin(3)] in body frame
            vang  = v6[:3]
            vlin  = np.clip(v6[3:], -10, 10)

            expmap = quat2expmap(q)
            ltype  = limb_type_vec(bname)

            if bname == "torso":
                angle = 0.0
                jrange = np.array([0.0, 0.0], dtype=np.float32)
            else:
                body_id = bid
                jnt_adr = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[jnt_adr]     # (본체당 1조인트 가정)
                angle = np.degrees(self.data.qpos[qpos_adr])
                jrange = np.degrees(self.model.jnt_range[jnt_adr]).astype(np.float32)
                angle = (angle - jrange[0]) / (jrange[1] - jrange[0] + 1e-8)
                jrange = (np.array([180.0, 180.0], dtype=np.float32) + jrange) / 360.0

            return np.concatenate([xpos, vlin, vang, expmap, ltype, [angle], jrange]).astype(np.float32)

        # mujoco-py의 model.body_names 대신 직접 이름 리스트 구성
        bodies = self._body_names()[1:]  # world 제외
        full = np.concatenate([obs_per_limb(b) for b in bodies])
        return full.ravel()
    # ----------------------------------------


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        # Gymnasium 렌더러와의 호환을 위해 안전 가드
        try:
            viewer = self.mujoco_renderer.viewer
            if viewer is not None:
                viewer.cam.trackbodyid = 2
                viewer.cam.distance = self.model.stat.extent * 0.75
                viewer.cam.lookat[2] = 1.15
                viewer.cam.elevation = -20
        except Exception:
            pass