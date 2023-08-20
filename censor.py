import networkx as nx
import numpy as np
from opinion_dynamics import *

class Censor:
    def __init__(self, bias, favored, discount_factor, T, strength, Nc):
        self.bias = bias 
        self.favored = favored
        self.gamma = discount_factor 
        self.T = T 
        self.strength = strength
        self.Nc = Nc                   # 单位时间审查次数


    def couple(self, society):
        self.society = society
        self.banned = np.zeros(self.society.network.number_of_nodes(), dtype=bool)
        self.time = np.zeros(shape=self.banned.shape)
        self.discount_factors = np.ones_like(self.time)
        self.neighbors = {}

    def update(self):
        if np.random.uniform(0, 1) < self.Nc * self.society.dt:
            p = p_banned(self.society.current_opinions(key='extrinsic'), self.strength, self.favored, self.bias)
            banned  = np.logical_or(self.banned, np.random.uniform(0, 1, size=p.shape) < p)
            new = np.logical_xor(self.banned, banned)
            self.banned = banned

            for i, flag in enumerate(new):
                if flag == True:
                    self.neighbors[i] = list(self.society.network.neighbors(i))
                    self.society.network.remove_edges_from(list(zip([i] * len(self.neighbors[i]), self.neighbors[i])))
                    
            self.discount_factors[new] *= self.gamma

        self.time[self.banned] += self.society.dt
        self.society.extrinsic_opinions[:, -1] *= self.discount_factors

        for i, t in enumerate(self.time):
            if t >= self.T:
                self.time[i] = 0
                self.banned[i] = False 
                self.society.network.add_edges_from(list(zip([i] * len(self.neighbors[i]), self.neighbors[i])))

def p_banned(x : np.array, strength, favored = 0, bias = 2):
    # probability being banned.  more extreme the opinion, more likely being banned
    # strength:  a parameter describing the strength of censorship
    # favored:   which side of the opinion are favored by censorship, thus less likely being banned  

    p = np.abs(np.tanh(strength * x))
    if favored == 0:
        return p
    else:
        positive = x > 0
        negative = x < 0

        if favored > 0:
            # positive opinions are favored, negative ones are banned
            p[positive] = p[positive] / bias
        else:
            p[negative] = p[negative] / bias
        
        return p