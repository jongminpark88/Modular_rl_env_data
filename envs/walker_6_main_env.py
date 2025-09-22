import os
from .modular_env import ModularEnv

XML_PATH = os.path.join(os.path.dirname(__file__), "..", "xmls", "walker_6_main.xml")

class Walker6MainEnv(ModularEnv):
    def __init__(self, frame_skip: int = 4, **kwargs):
        super().__init__(XML_PATH, frame_skip=frame_skip, **kwargs)
