import os
from .modular_env import ModularEnv

XML_PATH = os.path.join(os.path.dirname(__file__), "..", "xmls", "cheetah_3_balanced.xml")

class Cheetah3BalancedEnv(ModularEnv):
    def __init__(self, frame_skip: int = 4, **kwargs):
        super().__init__(XML_PATH, frame_skip=frame_skip, **kwargs)
