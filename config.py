# config.py
import os

# 루트 기준 상대 경로
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

XML_DIR  = os.path.join(ROOT_DIR, "xmls_converted")
ENV_DIR  = os.path.join(ROOT_DIR, "envs")

# envs/modular_env.py 의 템플릿 파일 경로
BASE_MODULAR_ENV_PATH = os.path.join(ENV_DIR, "modular_env.py")

# 수집된 전이 저장 경로
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
