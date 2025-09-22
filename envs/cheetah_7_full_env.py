import os
from .modular_env import ModularEnv

XML_PATH = os.path.join(os.path.dirname(__file__), "..", "xmls", "cheetah_7_full.xml")

class Cheetah7FullEnv(ModularEnv):
    def __init__(self, frame_skip: int = 4, **kwargs):
        super().__init__(XML_PATH, frame_skip=frame_skip, **kwargs)
