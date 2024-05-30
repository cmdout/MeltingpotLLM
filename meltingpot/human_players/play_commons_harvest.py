# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple human player for testing `commons_harvest`.

Use `WASD` keys to move the character around.
Use `Q and E` to turn the character.
Use `SPACE` to fire the zapper.
Use `TAB` to switch between players.
"""

import argparse
import json

from meltingpot.configs.substrates import commons_harvest__closed
from meltingpot.configs.substrates import commons_harvest__open
from meltingpot.configs.substrates import commons_harvest__partnership
from meltingpot.human_players import level_playing_utils
from ml_collections import config_dict
from typing import Dict, Any


environment_configs = {
    'commons_harvest__closed': commons_harvest__closed,
    'commons_harvest__open': commons_harvest__open,
    'commons_harvest__partnership': commons_harvest__partnership,
}

_ACTION_MAP = {
    'move': level_playing_utils.get_direction_pressed,
    'turn': level_playing_utils.get_turn_pressed,
    'fireZap': level_playing_utils.get_space_key_pressed,
}


def verbose_fn(unused_env, unused_player_index, unused_current_player_index):
    pass


def change_avatars_appearance(lab2d_settings: Dict[str, Any], is_focal_player: list[bool]):
    """
    Change the avatars appearance in the game environment

    Args:
        lab2d_settings: The lab2d settings for the game environment
        is_focal_player: List with the focal players
    Returns:
        A dictionary with the overrided configurations
    """
    new_color = (0, 0, 0, 255)  # Example new color
    game_objects = lab2d_settings['simulation']['gameObjects']

    for i in range(len(is_focal_player)):
        if not is_focal_player[i]:

            components = game_objects[i]['components']
            # Find the Appearance component
            for j, component in enumerate(components):
                if component.get('component') == 'Appearance':
                    # Override the first color ('!')
                    component['kwargs']['palettes'][0]['!'] = new_color
                    component['kwargs']['palettes'][0]['#'] = new_color
                    component['kwargs']['palettes'][0]['%'] = new_color
                    component['kwargs']['palettes'][0]['&'] = new_color
                    components[j] = component
                    break
            game_objects[i]['components'] = components

    overrided_configs = {'simulation': lab2d_settings['simulation']}
    overrided_configs['simulation']['gameObjects'] = game_objects

    return overrided_configs

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--level_name', type=str, default='commons_harvest__open',
        choices=environment_configs.keys(),
        help='Level name to load')
    parser.add_argument(
        '--observation', type=str, default='WORLD.RGB', help='Observation to render')
    parser.add_argument(
        '--settings', type=json.loads, default={}, help='Settings as JSON string')
    # Activate verbose mode with --verbose=True.
    parser.add_argument(
        '--verbose', type=bool, default=False, help='Print debug information')
    # Activate events printing mode with --print_events=True.
    parser.add_argument(
        '--print_events', type=bool, default=False, help='Print events')

    args = parser.parse_args()
    env_module = environment_configs[args.level_name]
    players = ['Laura', 'Juan', 'Pedro']
    env_config = env_module.get_config()
    is_focal_player = [True for _ in players]
    env_config = env_module.get_config_player(players)
    env_config.is_focal_player = is_focal_player
    with config_dict.ConfigDict(env_config).unlocked() as env_config:
        roles = env_config.default_player_roles
        env_config.lab2d_settings = env_module.build(roles, env_config)
        env_config.is_focal_player = is_focal_player
    config_overrides = change_avatars_appearance(env_config.lab2d_settings, is_focal_player)
    # stm = ShortTermMemory(agent_context_file=agent_context_file, world_context_file=world_context_file)
    # spatial_memory = SpatialMemory(scenario_map=scenario_info['scenario_map'],
    #                                     scenario_obstacles=scenario_info['scenario_obstacles'])
    level_playing_utils.run_episode(
        args.observation, config_overrides, _ACTION_MAP,
        env_config, level_playing_utils.RenderType.PYGAME,
        player_prefixes=players,
        verbose_fn=verbose_fn if args.verbose else None,
        print_events=args.print_events)


if __name__ == '__main__':
    main()
