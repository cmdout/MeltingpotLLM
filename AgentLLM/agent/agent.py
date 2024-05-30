import copy
import logging
import os
from queue import Queue
from typing import Union, Literal, Mapping, Callable

from AgentLLM.agent.cognitive_modules.perceive import create_memory, update_known_agents, update_known_objects, \
    should_react
from AgentLLM.agent.memory_structure.long_term_memory import LongTermMemory
from AgentLLM.agent.memory_structure.short_term_memory import ShortTermMemory
from AgentLLM.agent.memory_structure.spatial_memory import SpatialMemory
from AgentLLM.env.scene_descriptor.observations_generator import ObservationsGenerator
from AgentLLM.env.scene_descriptor.scene_descriptor import SceneDescriptor
from AgentLLM.utils.env_utils import check_agent_out_of_game
from AgentLLM.utils.logging import CustomAdapter
import dmlab2d

from AgentLLM.utils.queue_utils import list_from_queue

Mode = Union[Literal['normal'], Literal['cooperative']]
PLAYER_STR_FORMAT = 'player_{index}'
ActionMap = Mapping[str, Callable[[], int]]


class Agent:
    """Agent class.
    """

    def __init__(self, name: str, data_folder: str, agent_context_file: str, world_context_file: str,
                 scenario_info: dict, att_bandwidth: int = 10, reflection_umbral: int = 30, mode: Mode = 'normal',
                 understanding_umbral=30, observations_poignancy=10, prompts_folder="base_prompts_v0",
                 substrate_name="commons_harvest_open", start_from_scene=None, env_config=None, env_module=None) -> None:
        """Initializes the agent.

        Args:
            name (str): Name of the agent.
            data_folder (str): Path to the data folder.
            agent_context_file (str): Path to the json agent context file. Initial info about the agent.
            world_context_file (str): Path to the text world context file. Info about the world that the agent have access to.
            scenario_info (dict): Dictionary with the scenario info. Contains the scenario map and the scenario obstacles.
            att_bandwidth (int, optional): Attention bandwidth. The attention bandwidth is the number of observations that the agent can attend to at the same time. Defaults to 10.
            reflection_umbral (int, optional): Reflection umbral. The reflection umbral is the number of poignancy that the agent needs to accumulate to reflect on its observations. Defaults to 30.
            mode (Mode, optional): Defines the type of architecture to use. Defaults to 'normal'.
            understanding_umbral (int, optional): Understanding umbral. The understanding umbral is the number of poignancy that the agent needs to accumulate to update its understanding (only the poignancy of reflections are taken in account). Defaults to 6.
            observations_poignancy (int, optional): Poignancy of the observations. Defaults to 10.
            prompts_folder (str, optional): Folder where the prompts are stored. Defaults to "base_prompts_v0".
            substrate_name (str, optional): Name of the substrate. Defaults to "commons_harvest_open".
        """
        self.logger = logging.getLogger(__name__)
        self.logger = CustomAdapter(self.logger)

        self.name = name
        self.mode = mode
        self.att_bandwidth = att_bandwidth
        self.reflection_umbral = reflection_umbral
        self.observations_poignancy = observations_poignancy
        ltm_folder = os.path.join(data_folder, 'ltm_database')
        self.ltm = LongTermMemory(agent_name=name, data_folder=ltm_folder)
        # self.ltm = None
        self.stm = ShortTermMemory(agent_context_file=agent_context_file, world_context_file=world_context_file)
        self.spatial_memory = SpatialMemory(scenario_map=scenario_info['scenario_map'],
                                            scenario_obstacles=scenario_info['scenario_obstacles'])
        self.att_bandwidth = att_bandwidth
        self.understanding_umbral = understanding_umbral
        self.prompts_folder = prompts_folder
        self.stm.add_memory(memory=self.name, key='name')
        self.substrate_name = substrate_name
        self.action_map = ActionMap
        self.descriptor = SceneDescriptor(env_config)
        self.observationsGenerator = ObservationsGenerator(env_module.ASCII_MAP, env_config['player_names'], substrate_name)


        # Initialize steps sequence in empty queue
        self.stm.add_memory(memory=Queue(), key='current_steps_sequence')
        self.stm.add_memory(memory=scenario_info['valid_actions'], key='valid_actions')
        self.stm.add_memory(
            memory=f"{self.name}'s bio: {self.stm.get_memory('bio')} \nImportant: make all your decisions taking into "
                   f"account {self.name}'s bio." if self.stm.get_memory('bio') else "", key='bio_str')
        self.stm.add_memory(memory="You have not performed any actions yet.", key='previous_actions')

        if start_from_scene:
            self.ltm.load_memories_from_scene(scene_path=start_from_scene, agent_name=name)
            self.stm.load_memories_from_scene(scene_path=start_from_scene, agent_name=name)

    def move(self, observations, scene_description, state_changes, game_time, agent_reward):
        self.spatial_memory.update_current_scene(scene_description['global_position'],
                                                 scene_description['orientation'],
                                                 scene_description['observation'])
        react, filtered_observations, state_changes = self.perceive(observations, state_changes, game_time, agent_reward)

    def perceive(self, observations: list[str], changes_in_state: list[tuple[str, str]], game_time: str, reward: float,
                 is_agent_out: bool = False):
        """Perceives the environment and stores the observation in the long term memory. Decide if the agent should react to the observation.
        It also filters the observations to only store the closest ones, and asign a poignancy to the observations.
        Game time is also stored in the short term memory.
        Args:
            observations (list[str]): List of observations of the environment.
            game_time (str): Current game time.
            reward (float): Current reward of the agent.
            is_agent_out (bool, optional): True if the agent is out of the scenario (was taken), False otherwise. Defaults to False.

        Returns:
            tuple[bool, list[str], list[str]]: Tuple with True if the agent should react to the observation, False otherwise, the filtered observations and the changes in the state of the environment.
        """
        action_executed = self.stm.get_memory('current_action')
        if is_agent_out:
            memory = create_memory(self.name, game_time, action_executed, [], reward, observations,
                                   self.spatial_memory.position, self.spatial_memory.get_orientation_name(), True)
            self.ltm.add_memory(memory, game_time, self.observations_poignancy, {'type': 'perception'})
            current_observation = '\n'.join(observations)
            self.stm.add_memory(current_observation, 'current_observation')
            return False, observations, changes_in_state

        # Add the game time to the short term memory
        self.stm.add_memory(game_time, 'game_time')
        # Observations are filtered to only store the closest ones. The att_bandwidth defines the number of observations that the agent can attend to at the same time
        sorted_observations = self.spatial_memory.sort_observations_by_distance(observations)
        observations = sorted_observations[:self.att_bandwidth]

        # Update the agent known agents
        update_known_agents(observations, self.stm)
        # Update the agent known objects
        update_known_objects(observations, self.stm, self.substrate_name)

        # Parse the changes in the state of the environment observed by the agent
        changes = []
        for change, obs_time in changes_in_state:
            changes.append(f'{change} At {obs_time}')

        # Create a memory from the observations, the changes in the state of the environment and the reward, and add it to the long term memory
        position = self.spatial_memory.position
        orientation = self.spatial_memory.get_orientation_name()
        memory = create_memory(self.name, game_time, action_executed, changes, reward, observations, position,
                               orientation)
        self.ltm.add_memory(memory, game_time, self.observations_poignancy, {'type': 'perception'})

        current_observation = '\n'.join(observations)
        self.stm.add_memory(current_observation, 'current_observation')
        self.stm.add_memory(changes, 'changes_in_state')

        last_reward = self.stm.get_memory('current_reward') or 0.0
        self.stm.add_memory(reward, 'current_reward')
        self.stm.add_memory(last_reward, 'last_reward')
        last_position = self.stm.get_memory('current_position') or self.spatial_memory.position
        self.stm.add_memory(self.spatial_memory.position, 'current_position')
        self.stm.add_memory(last_position, 'last_position')
        self.stm.add_memory(orientation, 'current_orientation')

        # Decide if the agent should react to the observation
        current_plan = self.stm.get_memory('current_plan')
        world_context = self.stm.get_memory('world_context')
        agent_bio_str = self.stm.get_memory('bio_str')
        actions_sequence = list_from_queue(copy.copy(self.stm.get_memory('actions_sequence')))
        react, reasoning = should_react(self.name, world_context, observations, current_plan, actions_sequence, changes,
                                        game_time, agent_bio_str, self.prompts_folder)
        self.stm.add_memory(reasoning, 'reason_to_react')
        self.logger.info(f'{self.name} should react to the observation: {react}')
        return react, observations, changes



