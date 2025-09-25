# utils.py
from __future__ import annotations
import os
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from shutil import copyfile
import xmltodict

from config import XML_DIR, ENV_DIR, BASE_MODULAR_ENV_PATH
import wrappers


def makeEnvWrapper(env_name, obs_max_len=None, seed=0):
    """
    AsyncVectorEnv에서 콜러블로 쓰는 래퍼 팩토리 (Gymnasium)
    """
    def helper():
        e = gym.make(f"envs:{env_name}-v0")
        # Gymnasium: reset에서 seed 전달
        obs, info = e.reset(seed=seed)
        return wrappers.ModularEnvWrapper(e, obs_max_len)
    return helper


def findMaxChildren(env_names, graphs):
    max_children = 0
    for name in env_names:
        most_frequent = max(graphs[name], key=graphs[name].count)
        max_children = max(max_children, graphs[name].count(most_frequent))
    return max_children


def registerEnvs(env_names, max_episode_steps, custom_xml):
    """
    MuJoCo env를 Gymnasium에 등록하고, per-limb obs 크기와 max_action을 반환
    (모든 env에서 동일하다는 전제)
    """
    paths_to_register = []
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, f"{name}.xml"))
    else:
        if os.path.isfile(custom_xml):
            paths_to_register.append(custom_xml)
        elif os.path.isdir(custom_xml):
            for name in sorted(os.listdir(custom_xml)):
                if name.endswith(".xml"):
                    paths_to_register.append(os.path.join(custom_xml, name))

    limb_obs_size, max_action = None, None
    for xml in paths_to_register:
        env_name = os.path.basename(xml)[:-4]
        env_file = env_name  # envs/{env_file}.py 에 ModularEnv 있어야 함

        # env 파일이 없으면 템플릿 복사
        dst_py = os.path.join(ENV_DIR, f"{env_name}.py")
        if not os.path.exists(dst_py):
            copyfile(BASE_MODULAR_ENV_PATH, dst_py)

        params = {'xml': os.path.abspath(xml)}

        register(
            id=f"envs:{env_name}-v0",
            entry_point=f"envs.{env_file}:ModularEnv",
            max_episode_steps=max_episode_steps,
            kwargs=params
        )

        # limb_obs_size, max_action 추출 (Gymnasium)
        env = wrappers.IdentityWrapper(gym.make(f"envs:{env_name}-v0"))
        # Gymnasium reset tuple
        obs, info = env.reset()
        limb_obs_size = env.limb_obs_size
        max_action = env.max_action

    return limb_obs_size, max_action


def quat2expmap(q):
    if (np.abs(np.linalg.norm(q)-1) > 1e-3):
        raise ValueError("quat2expmap: input quaternion is not norm 1")
    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2*np.pi, 2*np.pi)
    if theta > np.pi:
        theta =  2 * np.pi - theta
        r0    = -r0
    r = r0 * theta
    return r


class ReplayBuffer(object):
    def __init__(self, max_size=1e6, slicing_size=None):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0
        self.slicing_size = slicing_size

    def add(self, data):
        if self.slicing_size is None:
            self.slicing_size = [data[0].size, data[1].size, data[2].size, 1, 1]
        arr = np.concatenate([data[0], data[1], data[2], [data[3]], [data[4]]])
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = arr
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(arr)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        s = self.slicing_size
        for i in ind:
            data = self.storage[i]
            X = data[:s[0]]
            Y = data[s[0]:s[0]+s[1]]
            U = data[s[0]+s[1]:s[0]+s[1]+s[2]]
            R = data[s[0]+s[1]+s[2]:s[0]+s[1]+s[2]+s[3]]
            D = data[s[0]+s[1]+s[2]+s[3]:]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
        return (np.array(x), np.array(y), np.array(u),
                np.array(r).reshape(-1,1), np.array(d).reshape(-1,1))


def getGraphStructure(xml_file):
    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch, self_idx)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    try:
        root = xml['mujoco']['worldbody']['body']
        assert not isinstance(root, list), 'worldbody must contain one torso body'
    except:
        raise Exception("XML not following standard MuJoCo format.")
    preorder(root)
    if 'walker' in os.path.basename(xml_file) and 'flipped' in os.path.basename(xml_file):
        parents[0] = -2
    return parents


def getGraphJoints(xml_file):
    def preorder(b):
        if 'joint' in b:
            if isinstance(b['joint'], list) and b['@name'] != 'torso':
                raise Exception("XML joints not standard.")
            elif not isinstance(b['joint'], list):
                b['joint'] = [b['joint']]
            joints.append([b['@name']])
            for j in b['joint']:
                joints[-1].append(j['@name'])
        if 'body' not in b:
            return
        if not isinstance(b['body'], list):
            b['body'] = [b['body']]
        for branch in b['body']:
            preorder(branch)
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml['mujoco']['worldbody']['body']
    except:
        raise Exception("XML not following standard MuJoCo format.")
    preorder(root)
    return joints


def getMotorJoints(xml_file):
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    motors = xml['mujoco']['actuator']['motor']
    if not isinstance(motors, list):
        motors = [motors]
    joints = [m['@joint'] for m in motors]
    return joints
