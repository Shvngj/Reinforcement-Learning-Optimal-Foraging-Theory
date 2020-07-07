import sys
from contextlib import closing
from io import StringIO
from gym import utils
import numpy as np
import math
from gym.envs.toy_text import discrete
from gym.utils import seeding
from gym import Env, spaces

#MAX_ROWS = 10
COLS = 3 #FIXED
#MAX_ATTEMPTS = 20
ACTIONS = 2

#MAP = ["+-----+"]+(["|B: :S|"]*MAX_ROWS)+["+-----+"]

class BushBerryEnv(discrete.DiscreteEnv):
    def __init__(self, MaximumRows = 10, MaximumAttempts = 13, ActionTime=2,TimeLag=3):
        self.MAX_ROWS = MaximumRows
        self.MAX_ATTEMPTS = MaximumAttempts
        self.ActionTime = ActionTime
        self.TimeLag = TimeLag
        
        MAP = ["+-----+"]+(["|B: :S|"]*self.MAX_ROWS)+["+-----+"]
        
        self.desc = np.asarray(MAP, dtype='c')
        num_states = self.MAX_ROWS*COLS*self.MAX_ATTEMPTS
        num_rows = self.MAX_ROWS
        num_columns = COLS
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = ACTIONS
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for attempt_idx in range(self.MAX_ATTEMPTS):
                    state = self.encode(row, col, attempt_idx)
                    initial_state_distrib[state] += 1
                    for action in range(num_actions):
                        new_row, new_col, new_attempt_idx = row, col, attempt_idx
                        done = False
                        if col == 0:
                            new_col = 1
                            new_attempt_idx+=1
                            reward = 10
                        elif col == 1:
                            if action == 0: #GoBush
                                new_col = 0
                                x = attempt_idx
                                reward =  7.434 - 0.5433626*x - 0.002967033*x*x
                                #reward = (7/self.ActionTime)*(1/(1+math.exp(1-attempt_idx))) 
                            elif action == 1: #GoBox
                                new_col = 2
                                reward = 7.434/(self.ActionTime+self.TimeLag)
                        elif col == 2:
                            new_row+=1
                            new_col=1
                            new_attempt_idx = 0
                            reward = 10
                        if row == self.MAX_ROWS - 1:
                            done = True

                        new_state = self.encode(
                            new_row, new_col, new_attempt_idx)
                        P[state][action].append(
                            (1.0, new_state, reward, done))   
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)
    
   
    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d= transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob" : p})
    
    
    def encode(self, row, col, attempt):
        return row*COLS*self.MAX_ATTEMPTS + col*self.MAX_ATTEMPTS + attempt
    
    def decode(self, i):
        a = (i % (COLS*self.MAX_ATTEMPTS)) % self.MAX_ATTEMPTS
        i -= a
        c = (i % (COLS*self.MAX_ATTEMPTS)) / self.MAX_ATTEMPTS
        i -= i % (COLS*self.MAX_ATTEMPTS)
        r = i/(COLS*self.MAX_ATTEMPTS)
        #assert 0 <= i < 5
        return [int(r),int(c),int(a)]
    
    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        row, col, attempt_idx = self.decode(self.s)
        out[1 + row][2 * col + 1] = utils.colorize(
                out[1 + row][2 * col + 1], 'green', highlight=True)
        outfile.write("\n".join(["".join(r) for r in out]) + "\n")
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
    
    def reset(self):
        self.s = self.encode(0,1,0)
        self.lastaction = None
        return self.s