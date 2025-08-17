from pursuer_strategies.astar_pp import astar_pursuer_policy
from pursuer_strategies.simple import simple_pursuer_policy
from evader_strategies.simple import simple_evader_policy
from evader_strategies.retreat import evader_retreat_policy
from evader_strategies.break_encircle import break_encircle_policy
from evader_strategies.astar_pp_evade import astar_evader_policy
from pursuer_strategies.PRM.route_selector import prm_pursuer_policy
from pursuer_strategies.ttr.ttr_pp import ttr_pursuer_policy
from pursuer_strategies.roleplay.roleplay import roleplay_pursuer_policy
import numpy as np

class pursuer_strategy_selector(): 
    def __init__(self):

        self.pursuer_strategy = {
            'astar': astar_pursuer_policy,
            'simple': simple_pursuer_policy,
            'prm': prm_pursuer_policy,
            'ttr': ttr_pursuer_policy,
            'roleplay': roleplay_pursuer_policy
        }

    def select_strategy(self, observation):
        """ 
        Selects the strategy based on the observation and name.
        Returns a dictionary of scores for each strategy.
        """

        return self.pursuer_strategy['prm'](observation)

class evader_strategy_selector():
    def __init__(self):
        self.evader_strategy = {
            'astar': astar_evader_policy,
            'simple': simple_evader_policy,
            'retreat': evader_retreat_policy,
            'break_encircle': break_encircle_policy
        }

    def select_strategy(self, observation):
        """
        Selects the strategy based on the observation and name.
        Returns a dictionary of scores for each strategy.
        """
        # return the astar strategy
        return self.evader_strategy['break_encircle'](observation)