# collectors/sac_collect.py
import os
import sys
import argparse
import shutil
from datetime import datetime
import csv
import json
import gzip
from typing import Optional

import numpy as np

# 프로젝트 루트 등록
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from config import XML_DIR  # xmls_converted 또는 네가 쓰는 XML 경로
from config import DATA_DIR
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.logger import configure as sb3_configure
from gymnasium.wrappers import TimeLimit
import importlib
import torch

from collectors.buffer import StreamingRecordingCallback  # 중요: 절대/네임스페이스 import


def make_env(env_module: str, xml_filename: Optional[str] = None, render_mode: Optional[str] = None):
    """envs.<env_module>.ModularEnv(xml=...) 를 생성한다."""
    mod = importlib.import_module(f"envs.{env_module}")
    EnvCls = getattr(mod, "ModularEnv")
    if xml_filename is None:
        xml_filename = os.path.join(XML_DIR, f"{env_module}.xml")
    env = EnvCls(xml=xml_filename, render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=2000)
    return env


def wrap_vec_env(env, log_dir: Optional[str] = None):
    """SB3에서 요구하는 Monitor 래핑 + DummyVecEnv (1 env)."""
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    else:
        env = Monitor(env)
    return DummyVecEnv([lambda: env])


def map_done_code_from_info(done_flag: bool, info: dict) -> int:
    """0=미종료, 1=terminated, 2=truncated(TimeLimit 등)"""
    if not done_flag:
        return 0
    if info.get("TimeLimit.truncated", False):
        return 2
    return 1


def _json_ser(x) -> str:
    return json.dumps(np.asarray(x).tolist(), separators=(",", ":"))


def collect_transitions_stream_csv(
    env,
    policy,
    out_path: str,
    total_steps: int,
    seed: Optional[int] = None,
    gzip_output: bool = True,
) -> str:
    """
    전문가/추론 롤아웃 전이를 CSV(.csv.gz)로 스트리밍 저장 (메모리 안전).
    """
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()

    if gzip_output and not out_path.endswith(".gz"):
        out_path = out_path + ".gz"

    if gzip_output:
        fh = gzip.open(out_path, "wt", newline="")
    else:
        fh = open(out_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["step", "state", "action", "next_state", "reward", "done_code", "time"])

    step = 0
    try:
        for _ in range(total_steps):
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done_code = 1 if terminated else (2 if truncated else 0)
            ns = info.get("terminal_observation", next_obs) if (terminated or truncated) else next_obs

            step += 1
            writer.writerow([
                step,
                _json_ser(obs),
                _json_ser(action),
                _json_ser(ns),
                float(reward),
                int(done_code),
                datetime.now().isoformat(timespec="seconds"),
            ])

            if (step % 2000) == 0:
                fh.flush()

            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs
    finally:
        fh.flush()
        fh.close()

    return out_path


def train_sac(env_name: str, total_timesteps: int, xml: Optional[str], seed: int, render: bool, device: str):
    """
    SAC 학습:
    - logs/<env_name>.csv
    - checkpoint/<env_name>_best.pt
    - data/<env_name>_*_train.csv.gz (전체 학습 전이; 스트리밍)
    - data/<env_name>_*_expert.csv.gz (베스트 정책 롤아웃 전이; 스트리밍)
    """
    ckpt_dir = os.path.join(DATA_DIR, "checkpoint")
    logs_dir = os.path.join(DATA_DIR, "log")
    data_dir = os.path.join(DATA_DIR, "data")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sb3_log_dir = os.path.join(logs_dir, f"{env_name}_{ts}")
    os.makedirs(sb3_log_dir, exist_ok=True)

    render_mode = "human" if render else None
    train_env = make_env(env_name, xml_filename=xml, render_mode=render_mode)
    eval_env = make_env(env_name, xml_filename=xml, render_mode=None)

    train_env_vec = wrap_vec_env(train_env, log_dir=sb3_log_dir)
    eval_env_vec = wrap_vec_env(eval_env, log_dir=None)

    logger = sb3_configure(sb3_log_dir, ["csv"])

    # 디바이스 표시
    print(f"[INFO] Using device: {device} (torch={torch.__version__})")

    model = SAC(
        policy="MlpPolicy",
        env=train_env_vec,
        verbose=1,
        tensorboard_log=None,
        seed=seed,
        device=device,  # MPS/CUDA/CPU
    )
    model.set_logger(logger)

    # 콜백: 평가 + CSV 스트리밍 기록
    best_tmp_dir = os.path.join(ckpt_dir, f"{env_name}_best_tmp")
    eval_cb = EvalCallback(
        eval_env_vec,
        best_model_save_path=best_tmp_dir,
        log_path=sb3_log_dir,
        eval_freq=max(10000, total_timesteps // 50),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )
    rec_cb = StreamingRecordingCallback(
        save_dir=data_dir,
        env_name=env_name,
        xml_path=(xml or os.path.join(XML_DIR, f"{env_name}.xml")),
        total_steps_hint=total_timesteps,
        gzip_output=True,
        flush_every=2000,
        verbose=0,
    )
    callbacks = CallbackList([eval_cb, rec_cb])

    # 진행바 사용 (tqdm, rich 필요)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # 베스트 모델 백업
    best_zip = os.path.join(best_tmp_dir, "best_model.zip")
    if not os.path.exists(best_zip):
        model.save(best_zip)

    best_pt = os.path.join(ckpt_dir, f"{env_name}_best.pt")
    shutil.copyfile(best_zip, best_pt)
    print(f"[OK] Best checkpoint saved to: {best_pt}  (SB3 zip 포맷)")

    # 학습 로그 progress.csv를 고정 경로로 복사
    src_progress = os.path.join(sb3_log_dir, "progress.csv")
    dst_csv = os.path.join(logs_dir, f"{env_name}.csv")
    try:
        shutil.copyfile(src_progress, dst_csv)
        print(f"[OK] Logs saved to: {dst_csv}")
    except Exception as e:
        print(f"[WARN] Could not copy progress.csv -> {dst_csv}: {e}")

    # 학습 전이 CSV 경로 보고
    print(f"[OK] Training transitions streamed to: {rec_cb.out_path}")

    # --- 베스트 모델 로드 & 전문가 롤아웃을 CSV로 스트리밍 저장 ---
    best_model = SAC.load(best_zip, device=device)
    try:
        best_model.set_env(eval_env_vec)
    except Exception:
        pass

    rollout_env = make_env(env_name, xml_filename=xml, render_mode=None)
    expert_base = os.path.join(data_dir, f"{env_name}_{ts}_expert.csv")
    expert_csv_path = collect_transitions_stream_csv(
        rollout_env, best_model, out_path=expert_base,
        total_steps=total_timesteps, seed=seed, gzip_output=True
    )
    print(f"[OK] Expert transitions streamed: {expert_csv_path}")

    train_env_vec.close()
    eval_env_vec.close()
    rollout_env.close()


def inference_sac(env_name: str, xml: Optional[str], checkpoint_path: str, steps: int, seed: int, render: bool, device: str):
    """
    체크포인트(.pt/.zip)로 추론만 수행:
    - data/<env_name>_*_inference.csv.gz 에 전이 저장 (스트리밍)
    """
    data_dir = os.path.join(PROJ_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 지정 디바이스로 로드
    model = SAC.load(checkpoint_path, device=device)

    render_mode = "human" if render else None
    env = make_env(env_name, xml_filename=xml, render_mode=render_mode)

    # (선택) vec-env 세팅 시도
    try:
        model.set_env(wrap_vec_env(make_env(env_name, xml_filename=xml, render_mode=None)))
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(data_dir, f"{env_name}_{ts}_inference.csv")
    csv_path = collect_transitions_stream_csv(
        env, model, out_path=out_base, total_steps=steps, seed=seed, gzip_output=True
    )
    print(f"[OK] Inference transitions streamed: {csv_path}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train or Inference SAC (SB3) per environment and save checkpoint/logs/transitions")
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="train 또는 inference")
    parser.add_argument("--env", type=str, required=True, help="envs/<name>.py 의 <name> (예: hopper_4, cheetah_3_balanced, walker_5_main)")
    parser.add_argument("--xml", type=str, default=None, help="사용할 XML 경로 (미지정 시 config.XML_DIR/<env>.xml)")
    parser.add_argument("--steps", type=int, default=2_000_000, help="학습 혹은 추론 스텝 수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="렌더링(human)")
    parser.add_argument("--checkpoint", type=str, default=None, help="inference 모드에서 필수: 체크포인트(.pt/.zip)")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch 디바이스: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps' 등")
    args = parser.parse_args()

    # 디바이스 문자열만 SB3에 전달
    if args.mode == "train":
        train_sac(
            env_name=args.env,
            total_timesteps=args.steps,
            xml=args.xml,
            seed=args.seed,
            render=args.render,
            device=args.device
        )
    else:
        if not args.checkpoint:
            raise ValueError("inference 모드에서는 --checkpoint 가 필수입니다.")
        inference_sac(
            env_name=args.env,
            xml=args.xml,
            checkpoint_path=args.checkpoint,
            steps=args.steps,
            seed=args.seed,
            render=args.render,
            device=args.device,
        )


if __name__ == "__main__":
    main()
