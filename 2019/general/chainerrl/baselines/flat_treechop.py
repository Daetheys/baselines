from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.mc import MS_PER_STEP, STEPS_PER_MS
from minerl.herobraine.hero.handler import Handler
from typing import List

import minerl.herobraine
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_spec import EnvSpec

TREECHOP_DOC = """
"""
INFTY = 2147483647
class TreechopFlat(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLTreechopFlat-v0'

        self.__time = 0

        super().__init__(*args,
                         max_episode_steps=INFTY, reward_threshold=INFTY,
                         **kwargs)
        
    def _entry_point(self,fake=False):
        return "__main__:_SingleAgentEnv"

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForCollectingItems([
                dict(type="log", amount=1, reward=1.0),
            ])
        ]

    def create_agent_start(self) -> List[Handler]:
        return [
            handlers.SimpleInventoryAgentStart([
                dict(type="diamond_axe", quantity=1)
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return []

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromPossessingItem([
                dict(type="log", amount=INFTY)]
            )
        ]

    def create_server_world_generators(self) -> List[Handler]:
        return [
            #handlers.DefaultWorldGenerator(force_reset="true",generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS)
            handlers.FlatWorldGenerator(generatorString='1;7,2;4;decoration')
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        return [
            handlers.ServerQuitFromTimeUp(
                (INFTY)),
            handlers.ServerQuitWhenAnyAgentFinishes()
        ]

    def create_server_decorators(self) -> List[Handler]:
        return []

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=False
            )
        ]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'survivaltreechop'

    def get_docstring(self):
        return TREECHOP_DOC

try:
    env_abs = TreechopFlat()
    env_abs.register()
except gym.error.Error:
    pass
