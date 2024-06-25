import pickle
import random

import numpy as np
from ml_collections import config_dict

from AgentLLM.env.MeltingPotEnvLLM import MeltingPotEnvLLM
from meltingpot import substrate
from meltingpot.configs.substrates import clean_up, commons_harvest__closed, commons_harvest__open, \
    commons_harvest__partnership
from AgentLLM.agent.agent import Agent
from AgentLLM.utils.args_handler import get_args
from AgentLLM.utils.env_utils import get_defined_valid_actions, condition_to_end_game, generate_agent_actions_map
from AgentLLM.utils.logging import setup_logging
from datetime import datetime
import logging
import os
import time
import cv2
import traceback

from meltingpot.human_players.play_commons_harvest import change_avatars_appearance
from meltingpot.utils.substrates import builder
from utils.logging import setup_logging, CustomAdapter
from utils.queue_utils import new_empty_queue
from utils.args_handler import get_args
from utils.files import extract_players, persist_short_term_memories, create_directory_if_not_exists

# Set up logging timestamp
logger_timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

logger = logging.getLogger(__name__)
rounds_count = 0

environment_configs = {
    'clean_up': clean_up,
    'commons_harvest_closed': commons_harvest__closed,
    'commons_harvest_open': commons_harvest__open,
    'commons_harvest_partnership': commons_harvest__partnership,
}


def train_llm_agent(args, logger):
    scene_path = None
    exp_data_path = "data/experiment_data/" + args.substrate + '/' + logger_timestamp
    os.makedirs(exp_data_path, exist_ok=True)
    logger.info("Program started")
    if args.start_from_scene:
        scene_path = f"data/scenes/{args.start_from_scene}"
        os.system(f"cp {scene_path}/variables.txt config/start_variables.txt")

    # Define players
    experiment_path = os.path.join("data", "defined_experiments", args.substrate)
    agents_bio_dir = os.path.join(experiment_path, "agents_context", args.agents_bio_config)
    game_scenario = args.scenario if args.scenario != "default" else None
    players_context = [os.path.abspath(os.path.join(agents_bio_dir, player_file)) for player_file in
                       os.listdir(agents_bio_dir)]

    players = extract_players(players_context)

    world_context_path = os.path.join(experiment_path, "world_context", f'{args.world_context}.txt')
    valid_actions = get_defined_valid_actions(game_name=args.substrate)
    scenario_obstacles = ['W', '$']  # TODO : Change this. This should be also loaded from the scenario file
    env_module = environment_configs[args.substrate]
    scenario_info = {'scenario_map': env_module.ASCII_MAP, 'valid_actions': valid_actions,
                     'scenario_obstacles': scenario_obstacles}  ## TODO: ALL THIS HAVE TO BE LOADED USING SUBSTRATE NAME
    data_folder = "data" if not args.simulation_id else f"data/databases/{args.simulation_id}"
    create_directory_if_not_exists(data_folder)
    # Start the game server
    # env_config_ = env_module.get_config()
    is_focal_player = [True for _ in players]
    env_config = env_module.get_config_player(players)
    env_config.is_focal_player = is_focal_player
    with config_dict.ConfigDict(env_config).unlocked() as env_config:
        roles = env_config.default_player_roles
        env_config.lab2d_settings = env_module.build(roles, env_config)
        env_config.is_focal_player = is_focal_player
    config_overrides = change_avatars_appearance(env_config.lab2d_settings, is_focal_player)
    env_config.lab2d_settings.update(config_overrides)
    env = MeltingPotEnvLLM(env_module, env_config, args.substrate)
    # Create agents
    agents = [Agent(name=player, data_folder=data_folder, agent_context_file=player_context,
                    world_context_file=world_context_path, scenario_info=scenario_info, mode=args.mode,
                    prompts_folder=str(args.prompts_source), substrate_name=args.substrate, start_from_scene=scene_path,
                    env_config=env_config, env_module=env_module)
              for player, player_context in zip(players, players_context)]
    for player in players:
        os.makedirs(exp_data_path + '/' + player, exist_ok=True)
    logger = CustomAdapter(logger, game_env=env)
    # We are setting args.prompts_source as a global variable to be used in the LLMModels class
    try:
        train_loop(agents, args.substrate, args.persist_memories, env)
    except KeyboardInterrupt:
        logger.info("Program interrupted. %s rounds executed.", rounds_count)
    except Exception as e:
        logger.exception("Rounds executed: %s. Exception: %s", rounds_count, e)
    store_exp_data(exp_data_path, agents)


def train_loop(agents, substrate_name, persist_memories, env):
    rounds_count, steps_count, max_rounds = 0, 0, 100
    time_step = env.reset()
    env.render()

    frame_size = (env.render().shape[1], env.render().shape[0])
    out = cv2.VideoWriter(logger_timestamp + '_gameplay.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, frame_size)

    while rounds_count < max_rounds and condition_to_end_game(substrate_name, env.get_current_global_map()):
        actions = {player_name: env.default_agent_actions_map() for player_name in env.player_prefixes}
        for id, agent in enumerate(agents):
            agent.store_state(id, time_step)
            all_observations = env.get_observations_by_player(agent.name)
            observations = all_observations['curr_state']
            scene_description = all_observations['scene_description']
            state_changes = all_observations['state_changes']
            agent_reward = env.score[agent.name]
            game_time = env.get_time()
            step_actions = agent.move(observations, scene_description, state_changes, game_time, agent_reward)
            while not step_actions.empty():
                step_action = step_actions.get()
                # Update the actions map for the agent
                actions[agent.name] = generate_agent_actions_map(step_action, env.default_agent_actions_map())
                logger.info('Agent %s action map: %s', agent.name, actions[agent.name])
                time_step = env.step(actions)
                agent.store_actions(id, actions, time_step)
                if not step_actions.empty():
                    agent.store_state(id, time_step)
                frame = env.render()
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            actions[agent.name] = env.default_agent_actions_map()
            # memories = {agent.name: agent.stm.get_memories().copy() for agent in agents}
            # persist_short_term_memories(memories, rounds_count, steps_count, logger_timestamp)
        rounds_count += 1
    out.release()
    cv2.destroyAllWindows()


def store_exp_data(data_folder, agents):
    for agent in agents:
        data_folder_ = data_folder + '/' + agent.name
        np.save(data_folder_ + '/world_history.npy', np.array(agent.world_history))
        np.save(data_folder_ + '/obs_history.npy', np.array(agent.obs_history))
        np.save(data_folder_ + '/orientation_history', np.array(agent.orientation_history))
        np.save(data_folder_ + '/position_history', np.array(agent.position_history))
        np.save(data_folder_ + '/reward_history.npy', np.array(agent.reward_history))
        with open(data_folder_ + '/action_history.pkl', 'wb') as f:
            pickle.dump(agent.action_history, f)


if __name__ == "__main__":
    args = get_args()
    setup_logging(logger_timestamp)
    start_time = time.time()
    train_llm_agent(args, logger)

    # current_directory = os.getcwd()
    # video_path = os.path.join(current_directory, 'gameplay.avi')
    # print(f"Video saved at: {video_path}")
    # If the experiment is "personalized", prepare a start_variables.txt file on config path
    # It will be copied from args.scene_path, file is called variables.txt
