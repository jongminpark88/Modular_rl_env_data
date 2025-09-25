# collectors/buffer.py
import os
import numpy as np
import csv
import json
import gzip
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


def map_done_code_from_info(done_flag: bool, info: dict) -> int:
    """0=미종료, 1=terminated, 2=truncated(TimeLimit 등)"""
    if not done_flag:
        return 0
    if info.get("TimeLimit.truncated", False):
        return 2
    return 1


class StreamingRecordingCallback(BaseCallback):
    """
    메모리 누적 없이 전이를 CSV(.csv 또는 .csv.gz)로 스트리밍 저장.
    VecEnv 지원. 매 step마다 N개의 (env 수만큼) row를 기록.
    """
    def __init__(
        self,
        save_dir: str,
        env_name: str,
        xml_path: str,
        total_steps_hint: int = 0,
        gzip_output: bool = True,
        flush_every: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "csv.gz" if gzip_output else "csv"
        self.out_path = os.path.join(save_dir, f"{env_name}_{ts}_train.{ext}")
        self.env_name = env_name
        self.xml_path = xml_path
        self.total_steps_hint = total_steps_hint
        self.flush_every = int(flush_every)

        self._fh = None       # file handle
        self._writer = None   # csv.writer
        self._rows_written = 0
        self._step_counter = 0

    @staticmethod
    def _ser(arr: np.ndarray) -> str:
        """numpy 배열을 JSON 문자열로 직렬화 (가독성/호환성 우수)."""
        return json.dumps(np.asarray(arr).tolist(), separators=(",", ":"))

    def _open_writer(self):
        if self._fh is not None:
            return
        if self.out_path.endswith(".gz"):
            self._fh = gzip.open(self.out_path, mode="wt", newline="")
        else:
            self._fh = open(self.out_path, mode="w", newline="")

        self._writer = csv.writer(self._fh)
        # 헤더
        self._writer.writerow([
            "global_step",   # 학습 전체 step (VecEnv 스텝 횟수)
            "env_idx",       # 어떤 서브-환경인지
            "state",         # JSON 문자열
            "action",        # JSON 문자열
            "next_state",    # JSON 문자열
            "reward",        # float
            "done_code",     # 0=not done, 1=terminated, 2=truncated
            "info_time",     # ISO 시각
        ])

    def _close_writer(self):
        if self._fh:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._writer = None

    def _on_training_start(self) -> None:
        self._open_writer()
        return True

    def _on_rollout_start(self) -> None:
        return True

    def _on_step(self) -> bool:
        # SB3 locals
        actions = self.locals.get("actions", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        new_obs = self.locals.get("new_obs", None)

        # model이 마지막 obs 캐시를 들고 있음
        last_obs = getattr(self.model, "_last_obs", None)
        if last_obs is None:
            return True

        self._step_counter += 1
        n_envs = int(self.training_env.num_envs)

        for i in range(n_envs):
            info_i = infos[i] if isinstance(infos, (list, tuple)) else infos
            done_code = map_done_code_from_info(bool(dones[i]), info_i)

            # terminal_observation 있으면 그것을 next_state로
            if dones[i]:
                ns = np.array(info_i.get("terminal_observation", new_obs[i]), copy=False)
            else:
                ns = np.array(new_obs[i], copy=False)

            row = [
                self._step_counter,
                i,
                self._ser(last_obs[i]),
                self._ser(actions[i]),
                self._ser(ns),
                float(rewards[i]),
                int(done_code),
                datetime.now().isoformat(timespec="seconds"),
            ]
            self._writer.writerow(row)
            self._rows_written += 1

        if self._rows_written % self.flush_every == 0:
            self._fh.flush()

        return True

    def _on_training_end(self) -> None:
        # 파일 닫기
        self._close_writer()
        # 메타 정보 파일 저장 (압축 없이 .json)
        meta_path = self.out_path.replace("_train.", "_train_meta.")
        if meta_path.endswith(".gz"):
            meta_path = meta_path[:-3]
        meta = {
            "env": self.env_name,
            "xml": self.xml_path,
            "source": "sb3_sac_training_stream",
            "rows": self._rows_written,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return True
