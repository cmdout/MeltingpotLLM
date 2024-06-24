import numpy as np
from scipy.ndimage import label, center_of_mass
from collections import defaultdict

ASCII_MAP = None


def get_defined_valid_actions(game_name: str = 'commons_harvest_open'):
    if game_name == 'commons_harvest_open':
        return [
            'go to position (x,y): This action takes the agent to the position specified, if there is an apple in the position the apple would be taken. You can choose any position on the map from the top left [0, 0] to the bottom right [17, 23]',
            'immobilize player (player_name) at (x,y): This action takes the agent near the specified position and uses the light beam pointed to the specified position. If there is another agent in that position, the agent would be attacked and will be inactive for a few rounds, then it would be reinstanted on the game on another position.',
            'stay put: This action keep the agent in the same position.',
            'explore: This action makes the agent to explore the map, it moves to a random position on the observed portion of the map.',
        ]

    elif game_name == 'clean_up':
        return ['grab apple (x,y)',
                # 'attack player (player_name) at (x,y)',
                'explore',
                'clean dirt of river at (x,y)',
                'go to river bank at (x,y)',
                'go to apples field edge at (x,y)', ]


def condition_to_end_game(substrate_name: str, current_map: list[str]):
    """
    Check if the game has ended
    Args:
        substrate_name: Name of the game to run, the name must match a folder in game_environment/substrates/python
        current_map: The current map of the game
    Returns:
        A boolean indicating if the game has ended if condition for the specific substrate is met
    """

    if substrate_name == "commons_harvest_open":
        # Checks if there's any apple "A" in the current map
        for row in current_map:
            if "A" in row:
                return True

    return True


def generate_agent_actions_map(action: str, base_action_map: dict):
    """
    Description: Generates the action map for the agent

    Args:
        action (str): Action of the agent
        base_action_map (dict): Base action map for the agent

    Returns:
        dict: Action map for the agent
    """

    if len(action.split(' ')) == 1:
        if action == 'attack':
            base_action_map['fireZap'] = 1
        elif action == 'clean':
            base_action_map['fireClean'] = 1

    elif len(action.split(' ')) == 2:

        kind, dir = action.split(' ')
        int_dir = 0
        if kind == 'move':
            int_dir = 1 if dir == 'up' else 2 if dir == 'right' \
                else 3 if dir == 'down' else 4 if dir == 'left' \
                else 0
        elif kind == 'turn':
            int_dir = 1 if dir == 'right' else -1 if dir == 'left' else 0
        elif action == 'stay put':
            kind = 'move'
            int_dir = 0

        base_action_map[kind] = int_dir

    return base_action_map


def connected_elems_map(ascci_map: str | list[list[str]], elements_to_find):
    """
        Returns a dictionary with the connected components of the map and their elements

        Args:
            ascci_map (str | list[list[str]]): Map in ascci format
            elements_to_find (list): List of elements to find in the map

        Returns:
            dict: Dictionary with the connected components of the map and their elements
        """

    if isinstance(ascci_map, str):
        # Convierte la matriz en una matriz numpy
        matrix = np.array([list(row) for row in ascci_map.split('\n') if row != ''])
    else:
        matrix = np.array(ascci_map)

    # Generate mask
    mask = (matrix == elements_to_find[0])
    for elem in elements_to_find[1:]:
        mask |= (matrix == elem)

    # Encontrar componentes conectados
    labeled_matrix, num_features = label(mask)

    # Inicializa un diccionario para almacenar los centros de los componentes y sus elementos
    component_data = defaultdict(list)

    # Calcula el centro de cada componente y almacena sus elementos
    for i in range(1, num_features + 1):
        component_mask = labeled_matrix == i
        center = center_of_mass(component_mask)
        center_coords = (int(center[0]), int(center[1]))
        component_elements = np.argwhere(component_mask)
        component_data[i] = {'center': center_coords, 'elements': component_elements.tolist()}

    return dict(component_data)


def check_agent_out_of_game(observations: list[str]):
    """
     Description: Checks if the agent is out of the game

     Args:
         observations (list[str]): Observations of the agents

     Returns:
         bool: True if the agent is out of the game, False otherwise
    """
    return (len(observations) > 0 and observations[0].startswith(
        ('There are no observations: You were attacked', 'There are no observations: you\'re out of the game')))


def parse_string_to_matrix(input_string):
    rows = input_string.strip().split('\n')
    matrix = np.array([list(row) for row in rows])
    return matrix


def matrix_to_string(matrix):
    rows = [''.join(row) for row in matrix]
    return '\n'.join(rows)
