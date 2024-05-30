import collections
from copy import deepcopy
import datetime

import dm_env
import dmlab2d
import numpy as np
from gymnasium import spaces

from AgentLLM.env.scene_descriptor.observations_generator import ObservationsGenerator
from AgentLLM.env.scene_descriptor.scene_descriptor import SceneDescriptor
from AgentLLM.utils.env_utils import check_agent_out_of_game
from AgentLLM.utils.gym_utils import spec_to_space, remove_world_observations_from_space, default_agent_actions_map
from typing import Any, Callable, Mapping
from meltingpot.utils.substrates import builder

PLAYER_STR_FORMAT = 'player_{index}'
ActionMap = Mapping[str, Callable[[], int]]


class MeltingPotEnvLLM:
    """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

    def __init__(self, env_module, env_config, substrate_name):
        """Initializes the instance.

    Args:
      env: dmlab2d environment to wrap. Will be closed when this wrapper closes.
    """
        self.dateFormat = "%Y-%m-%d %H:%M:%S"
        self.time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        self.score = collections.defaultdict(float)
        self.curr_global_map = None
        self.curr_scene_description = None
        self.timestep = None
        self._env = builder.builder(**env_config)
        self._default_action = env_module.NOOP
        self._num_players = len(env_config['player_names'])
        self._ordered_agent_ids = [
            PLAYER_STR_FORMAT.format(index=index)
            for index in range(self._num_players)
        ]
        # RLLib requires environments to have the following member variables:
        # observation_space, action_space, and _agent_ids
        self._agent_ids = set(self._ordered_agent_ids)
        # RLLib expects a dictionary of agent_id to observation or action,
        # Melting Pot uses a tuple, so we convert
        self.time_step = None
        self.action_map = ActionMap
        self.descriptor = SceneDescriptor(env_config)
        self.observationsGenerator = ObservationsGenerator(env_module.ASCII_MAP, env_config['player_names'],
                                                           substrate_name)
        self.game_steps = 0  # Number of steps of the game
        self.player_prefixes = env_config['player_names']

    def reset(self, *args, **kwargs):
        """See base class."""
        self.timestep = self._env.reset()
        self.generate_observations()
        return self.timestep

    def step(self, current_actions_map):
        """Run one step of the game.

        Args:
            actions: A dictionary of actions for each player.
        Returns:
            A dictionary with the observations of each player.
        """
        self.game_steps += 1
        action_reader = ActionReader(self._env, self.action_map)
        agents_observing = []
        # Get the agents that are observing and didn't move
        game_actions = action_reader.various_agents_step(current_actions_map, self.player_prefixes)
        self.timestep = self._env.step(game_actions)
        self.generate_observations(current_actions_map)
        return self.timestep

    def close(self):
        """See base class."""
        self._env.close()

    def get_dmlab2d_env(self):
        """Returns the underlying DM Lab2D environment."""
        return self._env

    # Metadata is required by the gym `Env` class that we are extending, to show
    # which modes the `render` method supports.
    metadata = {'render.modes': ['rgb_array']}

    def render(self) -> np.ndarray:
        """Render the environment.

    This allows you to set `record_env` in your training config, to record
    videos of gameplay.

    Returns:
        np.ndarray: This returns a numpy.ndarray with shape (x, y, 3),
        representing RGB values for an x-by-y pixel image, suitable for turning
        into a video.
    """
        observation = self._env.observation()
        world_rgb = observation[0]['WORLD.RGB']

        # RGB mode is used for recording videos
        return world_rgb

    def _convert_spaces_tuple_to_dict(
            self,
            input_tuple: spaces.Tuple,
            remove_world_observations: bool = False) -> spaces.Dict:
        """Returns spaces tuple converted to a dictionary.

    Args:
      input_tuple: tuple to convert.
      remove_world_observations: If True will remove non-player observations.
    """
        return spaces.Dict({
            agent_id: (remove_world_observations_from_space(input_tuple[i])
                       if remove_world_observations else input_tuple[i])
            for i, agent_id in enumerate(self._ordered_agent_ids)
        })

    def default_agent_actions_map(self):
        """
        Description: Returns the base action map for the agent
        Retrieves the action map from the game environment
        """
        return deepcopy(self._default_action)

    def get_observations_by_player(self, player_prefix):
        """Returns the observations of the given player.
        Args:
            player_prefix: The prefix of the player
        Returns:
            A dictionary with the observations of the player
        """
        curr_state = \
            self.observationsGenerator.get_all_observations_descriptions(str(self.curr_scene_description).strip())[
                player_prefix]
        scene_description = self.curr_scene_description[player_prefix]
        if check_agent_out_of_game(curr_state):
            state_changes = []
        else:
            # When the agent is out, do not get the state changes to accumulate them until the agent is revived
            state_changes = self.observationsGenerator.get_observed_changes_per_agent(player_prefix)
        return {
            'curr_state': curr_state,
            'scene_description': scene_description,
            'state_changes': state_changes
        }

    def generate_observations(self, current_actions_map=None):
        agents_observing = []
        # Get the agents that are observing and didn't move
        if current_actions_map:
            agents_observing = [agent_name for agent_name, action_map in current_actions_map.items() if
                                action_map == default_agent_actions_map(self.substrate_name)]

        description, curr_global_map = self.descriptor.describe_scene(self.timestep)
        rewards = _get_rewards(self.timestep)
        for i, prefix in enumerate(self.player_prefixes):
            self.score[prefix] += rewards[str(i + 1)]
        # Get the raw observations from the environment after the actions are executed

        # Update the observations generator
        game_time = self.get_time()
        self.observationsGenerator.update_state_changes(description, agents_observing, game_time)

        self.curr_scene_description = description
        self.curr_global_map = curr_global_map
        
    def get_current_global_map(self) -> dict:
        """Returns the current scene description."""
        return self.curr_global_map

    def get_time(self) -> str:
        """Returns the current time of the game. The time will be formatted as specified in the config file."""
        return self.time.strftime(self.dateFormat)

    def get_current_step_number(self) -> int:
        """Returns the current step number of the game."""
        return self.game_steps


class ActionReader(object):
    """Convert keyboard actions to environment actions."""

    def __init__(self, env: dmlab2d.Environment, action_map: ActionMap):
        # Actions are named "<player_prefix>.<action_name>"
        self._action_map = action_map
        self._action_spec = env.action_spec()
        assert isinstance(self._action_spec, dict)
        self._action_names = set()
        for action_key in self._action_spec.keys():
            _, action_name = _split_key(action_key)
            self._action_names.add(action_name)

    def step(self, player_prefix: str) -> Mapping[str, int]:
        """Update the actions of player `player_prefix`."""
        actions = {action_key: 0 for action_key in self._action_spec.keys()}
        for action_name in self._action_names:
            actions[f'{player_prefix}.{action_name}'] = self._action_map[
                action_name]()
        return actions

    def various_agents_step(self, new_action_map, player_prefixes) -> Mapping[str, int]:
        """Update the actions of player `player_prefix`.
        Args:
            new_action_map: A dictionary with the actions of each player. Keys are player prefixes
            player_prefixes: A list with the player prefixes
        Returns:
            A dictionary with the actions of each player. Keys are combination of player indices starting from 1 and action names
        """
        actions = {action_key: 0 for action_key in self._action_spec.keys()}
        for i, player_prefix in enumerate(player_prefixes):
            for action_name in self._action_names:
                actions[f'{i + 1}.{action_name}'] = new_action_map[player_prefix][action_name]
        return actions


def _split_key(key: str) -> tuple[str, ...]:
    """Splits the key into player index and name."""
    return tuple(key.split('.', maxsplit=1))


def _get_rewards(timestep: dm_env.TimeStep) -> Mapping[str, float]:
    """Gets the list of rewards, one for each player."""
    rewards = {}
    for key in timestep.observation.keys():
        if key.endswith('.REWARD'):
            player_prefix, name = _split_key(key)
            if name == 'REWARD':
                rewards[player_prefix] = timestep.observation[key]
    return rewards
