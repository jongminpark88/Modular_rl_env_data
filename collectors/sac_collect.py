# collectors/sac_collect.py
from __future__ import annotations
import os, time, numpy as np, torch
from typing import List, Dict
import gymnasium as gym

import utils  # registerEnvs, makeEnvWrapper, getGraphStructure
from config import XML_DIR, DATA_DIR


def list_envs_from_xml(xml_dir: str, morphologies: List[str] | None = None):
    names, graphs = [], {}
    for fn in sorted(os.listdir(xml_dir)):
        if not fn.endswith(".xml"):
            continue
        if morphologies and not any(m in fn for m in morphologies):
            continue
        name = fn[:-4]
        names.append(name)
        graphs[name] = utils.getGraphStructure(os.path.join(xml_dir, fn))
    assert names, "No XMLs found in XML_DIR."
    return names, graphs


def build_vec_env(env_names: List[str], graphs: Dict[str, List[int]], limb_obs_size: int, seed: int):
    """
    Gymnasium AsyncVectorEnv
    - env_fns: [callable, ...] 각기 다른 morphology
    - 반환 obs/reward/done 등이 배치(ndarray) 형태
    """
    max_limbs = max(len(graphs[n]) for n in env_names)
    obs_max_len = max_limbs * limb_obs_size
    env_fns = [utils.makeEnvWrapper(n, obs_max_len, seed) for n in env_names]
    vec_env = gym.vector.AsyncVectorEnv(env_fns)  # (num_envs,) batch
    return vec_env, max_limbs


def slice_valid_obs(obs_padded_1d: np.ndarray, num_limbs: int, limb_obs_size: int) -> np.ndarray:
    return obs_padded_1d[: num_limbs * limb_obs_size].astype(np.float32, copy=False)


def collect_transitions_with_sac(
    sac_policy,                   # sac_policy.act(obs_valid)-> np.ndarray in [-1,1]^L
    out_dir: str = DATA_DIR,
    max_steps: int = 200_000,
    max_episode_steps: int = 1000,   # registerEnvs에 전달
    seed: int = 0,
    morphologies: List[str] | None = None,
    save_chunk: int = 20_000,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) XML → env list & graphs
    env_names, graphs = list_envs_from_xml(XML_DIR, morphologies)
    print(f"[collect] envs: {env_names}")

    # 2) register → limb_obs_size, max_action
    limb_obs_size, max_action = utils.registerEnvs(env_names, max_episode_steps, custom_xml=False)
    print(f"[collect] limb_obs_size={limb_obs_size}, max_action={max_action}")

    # 3) AsyncVectorEnv 생성
    env, max_limbs = build_vec_env(env_names, graphs, limb_obs_size, seed)
    np.random.seed(seed); torch.manual_seed(seed)

    # 4) reset (Gymnasium batched)
    obs_batch, info_batch = env.reset(seed=seed)  # obs_batch: (N, obs_max_len)

    # 5) 버퍼
    buf = {n: {k: [] for k in ["obs","act","rew","next_obs","done"]} for n in env_names}
    num_envs = len(env_names)

    def flush(force=False):
        for idx, n in enumerate(env_names):
            if not buf[n]["done"]:
                continue
            if not force and len(buf[n]["done"]) < save_chunk:
                continue
            ts = int(time.time())
            path = os.path.join(out_dir, f"{n}_{ts}_{len(buf[n]['done'])}.npz")
            np.savez(
                path,
                obs=np.asarray(buf[n]["obs"], dtype=np.float32),
                act=np.asarray(buf[n]["act"], dtype=np.float32),
                rew=np.asarray(buf[n]["rew"], dtype=np.float32),
                next_obs=np.asarray(buf[n]["next_obs"], dtype=np.float32),
                done=np.asarray(buf[n]["done"], dtype=np.float32),
            )
            for k in buf[n]: buf[n][k].clear()
            print(f"[flush] saved {path}")

    total = 0
    while total < max_steps:
        # 6) 각 env별 유효 obs → SAC → 액션 만들고 패딩(벡터env 입력은 (N, max_limbs))
        act_valid_list = []
        for i, name in enumerate(env_names):
            L = len(graphs[name])
            obs_valid = slice_valid_obs(obs_batch[i], L, limb_obs_size)
            a = sac_policy.act(obs_valid)             # [-1,1]^L
            a = np.clip(a, -1.0, 1.0) * max_action    # scale
            # 유효 행동 따로 저장해두고, 벡터env 입력용 패딩 만들어서 넣음
            act_valid_list.append(a.astype(np.float32))

        # 벡터env 입력(패딩)
        actions_batched = np.zeros((num_envs, max_limbs), dtype=np.float32)
        for i, a_valid in enumerate(act_valid_list):
            actions_batched[i, :a_valid.size] = a_valid

        # 7) step (Gymnasium batched)
        next_obs_batch, rew_batch, terminated_batch, truncated_batch, info_batch = env.step(actions_batched)
        done_batch = np.logical_or(terminated_batch, truncated_batch)

        # 8) 전이 저장(개별 env별)
        for i, name in enumerate(env_names):
            L = len(graphs[name])
            obs_valid      = slice_valid_obs(obs_batch[i],      L, limb_obs_size)
            next_obs_valid = slice_valid_obs(next_obs_batch[i], L, limb_obs_size)
            a_valid        = act_valid_list[i]

            buf[name]["obs"].append(obs_valid)
            buf[name]["act"].append(a_valid)
            buf[name]["rew"].append(np.float32(rew_batch[i]))
            buf[name]["next_obs"].append(next_obs_valid)
            buf[name]["done"].append(np.float32(done_batch[i]))

        total += num_envs
        obs_batch = next_obs_batch
        flush(False)

    flush(True)
    env.close()
    print("[collect] done.")


# 예시용 더미 정책 (실사용시 policies.sac.Policy로 교체)
class DummySACPolicy:
    def __init__(self, limb_obs_size: int = 19):
        self.limb_obs_size = limb_obs_size
    def act(self, obs_valid: np.ndarray) -> np.ndarray:
        L = len(obs_valid) // self.limb_obs_size
        return np.random.uniform(-1.0, 1.0, size=(L,)).astype(np.float32)


if __name__ == "__main__":
    policy = DummySACPolicy(limb_obs_size=19)  # modular_env 기준 19
    collect_transitions_with_sac(
        sac_policy=policy,
        out_dir=DATA_DIR,
        max_steps=50_000,
        max_episode_steps=1000,
        seed=42,
        morphologies=None,
        save_chunk=10_000,
    )
