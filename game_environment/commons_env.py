import numpy as np

# from social_dilemmas.envs
from .commons_agent import HarvestCommonsAgent, HARVEST_DEFAULT_VIEW_SIZE
from .constants import HARVEST_MAP
from .map_env import MapEnv, ACTIONS

APPLE_RADIUS = 2

# Add custom actions to the agent
ACTIONS['FIRE'] = 5  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

OUTCAST_POSITION = -99

AGENT_COLOR = [181, 4, 10]  #
DEFAULT_COLORMAP = {' ': [0, 0, 0],  # Black background
                    '0': [0, 0, 0],  # Black background beyond map walls
                    '': [180, 180, 180],  # Grey board walls
                    '@': [180, 180, 180],  # Grey board walls
                    'A': [0, 255, 0],  # Green apples
                    'F': [255, 255, 0],  # Yellow fining beam
                    'P': [159, 67, 255],  # Purple player

                    # Colours for agents. R value is a unique identifier
                    '1': AGENT_COLOR,
                    '2': AGENT_COLOR,
                    '3': AGENT_COLOR,
                    '4': AGENT_COLOR,
                    '5': AGENT_COLOR,
                    '6': AGENT_COLOR,
                    '7': AGENT_COLOR,
                    '8': AGENT_COLOR,
                    '9': AGENT_COLOR,
                    '10': AGENT_COLOR,
                    }

MEDIUM_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A P   A    A    A    A  A    A    @',
    '@  AAA  AAA  AAA  AAA  AAAAAA  AAA   @',
    '@ A A    A    A    A    A  A    A   P@',
    '@PA             A      A       A     @',
    '@ A   A    A    A    A  A A  A    A  @',
    '@PAA AAA  AAA  AAA  AAA     AAA  AAA @',
    '@ A   A    A  A A  A A   P   A    A  @',
    '@PA                                P @',
    '@ A    A    A    A    A  A    A    A @',
    '@AAA  AAA  AAA  AAA  AA AAA  AAA  AA @',
    '@ A    A    A    A    A  A    A    A @',
    '@P A A A               P             @',
    '@P  A    A    A    A       P     P   @',
    '@  AAA  AAA  AAA  AAA         P    P @',
    '@P  A    A    A    A   P   P  P  P   @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

SMALL_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A    A    A    A  P AP@',
    '@PAAA  AAA  AAA  AAA  AAA@',
    '@  A    A    A    A    A @',
    '@P                       @',
    '@    A    A    A    A    @',
    '@   AAA  AAA  AAA  AAA   @',
    '@P P A    A    A    A P P@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

MAP = {"small": SMALL_HARVEST_MAP,
       "medium": MEDIUM_HARVEST_MAP}


class HarvestCommonsEnv(MapEnv):

    def __init__(self, ascii_map=HARVEST_MAP, num_agents=1, render=False, agent_view_range=HARVEST_DEFAULT_VIEW_SIZE,
                 color_map=None):
        if color_map is None:
            color_map = DEFAULT_COLORMAP
        self.apple_points = []
        self.agent_view_range = agent_view_range

        if color_map is None:
            color_map = DEFAULT_COLORMAP

        super().__init__(ascii_map, num_agents, render, color_map=color_map)

        self.rewards_record = {}
        self.timeout_record = {}

        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.apple_points.append([row, col])

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent = HarvestCommonsAgent(agent_id, spawn_point, rotation, grid, lateral_view_range=self.agent_view_range,
                                        frontal_view_range=self.agent_view_range)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.world_map[apple_point[0], apple_point[1]] = 'A'

        # reset social metrics
        self.rewards_record = {}
        self.timeout_record = {}

    def custom_action(self, agent, action):
        agent.fire_beam('F')
        updates = self.update_map_fire(agent.get_pos().tolist(),
                                       agent.get_orientation(),
                                       ACTIONS['FIRE'], fire_char='F')
        return updates

    def custom_map_update(self):
        "See parent class"
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)

        # Outcast timed-out agents
        for agent_id, agent in self.agents.items():
            if agent.remaining_timeout > 0:
                agent.remaining_timeout -= 1
                # print("Agent %s its on timeout for %d n_steps" % (agent_id, agent.remaining_timeout))
                if not np.any(agent.pos == OUTCAST_POSITION):
                    self.update_map([[agent.pos[0], agent.pos[1], ' ']])
                    agent.pos = np.array([OUTCAST_POSITION, OUTCAST_POSITION])
            # Return agent to environment
            if agent.remaining_timeout == 0 and np.any(agent.pos == OUTCAST_POSITION):
                # print("%s has finished timeout" % agent_id)
                spawn_point = self.spawn_point()
                spawn_rotation = self.spawn_rotation()
                agent.update_agent_pos(spawn_point)
                agent.update_agent_rot(spawn_rotation)

    def spawn_apples(self):
        """Construct the apples spawned in this step.

        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in self.agent_pos and self.world_map[row, col] != 'A':
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if 0 <= x + j < self.world_map.shape[0] and \
                                    self.world_map.shape[1] > y + k >= 0:
                                symbol = self.world_map[x + j, y + k]
                                if symbol == 'A':
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = np.random.rand(1)[0]
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, 'A'))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get('A', 0)
        return num_apples

    def update_social_metrics(self, rewards):
        # Save a record of rewards by agent as they are needed for the social metrics computation
        for agent_id, reward in rewards.items():
            if agent_id in self.rewards_record.keys():
                self.rewards_record[agent_id].append(reward)
            else:
                self.rewards_record[agent_id] = [reward]

            is_agent_in_timeout = True if self.agents[agent_id].remaining_timeout > 0 else False
            if agent_id in self.timeout_record.keys():
                self.timeout_record[agent_id].append(is_agent_in_timeout)
            else:
                self.timeout_record[agent_id] = [is_agent_in_timeout]

    def get_social_metrics(self, episode_steps):

        if len(self.rewards_record) < 1:
            return None

        # Compute sum of rewards
        sum_of_rewards = dict(zip(self.agents.keys(), [0] * self.num_agents))
        for agent_id, rewards in self.rewards_record.items():
            sum_of_rewards[agent_id] = np.sum(rewards)

        agents_sum_rewards = np.sum(list(sum_of_rewards.values()))

        # Compute efficiency/sustainability
        efficiency = agents_sum_rewards / self.num_agents

        # Compute Equality (Gini Coefficient)
        sum_of_diff = 0
        for agent_id_a, rewards_sum_a in sum_of_rewards.items():
            for agent_id_b, rewards_sum_b in sum_of_rewards.items():
                sum_of_diff += np.abs(rewards_sum_a - rewards_sum_b)

        equality = 1 - sum_of_diff / (2 * self.num_agents * agents_sum_rewards)

        # Compute sustainability metric (Average time of at which rewards were collected)
        avg_time = 0
        for agent_id, rewards in self.rewards_record.items():
            pos_reward_time_steps = np.argwhere(np.array(rewards) > 0)
            if pos_reward_time_steps.size != 0:
                avg_time += np.mean(pos_reward_time_steps)

        sustainability = avg_time / self.num_agents

        # Compute peace metric
        timeout_steps = 0
        for agent_id, peace_record in self.timeout_record.items():
            timeout_steps += np.sum(peace_record)
        peace = (self.num_agents * episode_steps - timeout_steps) / self.num_agents

        return efficiency, equality, sustainability, peace

