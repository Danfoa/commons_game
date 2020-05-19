"""Base class for an agent that defines the possible actions. """

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from .agent import BASE_ACTIONS, Agent

HARVEST_ACTIONS = BASE_ACTIONS.copy()
HARVEST_ACTIONS.update({7: 'FIRE'})  # Fire a penalty beam

HARVEST_DEFAULT_VIEW_SIZE = 5
TIMEOUT_TIME = 25


class HarvestCommonsAgent(Agent):

    def __init__(self, agent_id, start_pos, start_orientation, grid,
                 lateral_view_range=HARVEST_DEFAULT_VIEW_SIZE,
                 frontal_view_range=HARVEST_DEFAULT_VIEW_SIZE,):

        self.lateral_view_range = lateral_view_range
        self.frontal_view_range = frontal_view_range
        # When hit, agent is cast away from map for `remaining_timeout` n_steps
        self.remaining_timeout = 0

        super().__init__(agent_id, start_pos, start_orientation, grid,
                         row_size=self.frontal_view_range, col_size=self.frontal_view_range)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @property
    def action_space(self):
        return Discrete(8)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]

    @property
    def observation_space(self):
        return Box(low=0.0, high=0.0, shape=(2 * self.frontal_view_range + 1,
                                             2 * self.lateral_view_range + 1, 3), dtype=np.float32)

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 0
            if self.remaining_timeout == 0:
                # print("%s was hit with timeout beam" % self.agent_id)
                self.remaining_timeout = TIMEOUT_TIME

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 0

    def get_done(self):
        return False  # Agents never really reach a final state

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char

    def return_valid_pos(self, new_pos):
        if self.remaining_timeout > 0:
            return self.pos
        else:
            return super(HarvestCommonsAgent, self).return_valid_pos(new_pos)