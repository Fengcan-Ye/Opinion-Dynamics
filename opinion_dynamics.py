import networkx as nx
import numpy as np
from censor import *


class Society:
    def __init__(self, network : nx.Graph, 
                 K : float, 
                 alpha : float, 
                 beta : float,
                 dt : float):
        self.network = network 
        self.intrinsic_opinions, self.extrinsic_opinions = initialize_opinions(network.number_of_nodes())
        self.K = K 
        self.alpha = alpha 
        self.beta = beta 
        self.dt = dt
        self.coupled = False
    
    def current_opinions(self, key : str):
        if key.startswith('i'):
            return self.intrinsic_opinions[:, -1]
        elif key.startswith('e'):
            return self.extrinsic_opinions[:, -1]
        else:
            raise ValueError
    
    def couple(self, censor : Censor):
        self.censor = censor
        self.censor.couple(self)
        self.coupled = True
    
    def update(self):
        intrinsics = self.current_opinions(key='intrinsic opinion')
        extrinsics = self.current_opinions(key='extrinsic opinion')

        # P(i is influenced by j) ~ |I_i - E_j|^{-beta}  where j is a neighbor of i
        p_unnormalized = np.abs(intrinsics.reshape(-1, 1) - extrinsics) ** (-self.beta) * nx.adjacency_matrix(self.network)
        p_unnormalized = np.nan_to_num(p_unnormalized, nan=0, posinf=0, neginf=0)
        p = p_unnormalized / np.sum(p_unnormalized, axis=1).reshape(-1, 1)

        us = np.random.uniform(0, 1, size = p.shape)
        A = p > us
    
        intrinsics = intrinsics + \
                     RK4(intrinsics, 
                         lambda I: dxdt(I, extrinsics, A, self.K, self.alpha), 
                         self.dt)
        self.intrinsic_opinions = np.hstack([self.intrinsic_opinions, intrinsics.reshape(-1, 1)])
        self.extrinsic_opinions = np.hstack([self.extrinsic_opinions, intrinsics.reshape(-1, 1)])

        if self.coupled:
            self.censor.update()
    
    def trajectories(self):
        return self.intrinsic_opinions, self.extrinsic_opinions


def initialize_opinions(n):
    intrinsic_opinions = np.random.uniform(low=-1, high=1, size=(n, 1))
    return intrinsic_opinions, intrinsic_opinions.copy()

def RK4(x : np.array, f, h):
    # 4th order Runge-Kutta method
    # f does not explicitly involve t
    
    k1 = f(x)
    k2 = f(x + k1 * h / 2)
    k3 = f(x + k2 * h / 2)
    k4 = f(x + h * k3)

    return h / 6 * (k1 + 2*k2 + 2*k3 + k4)

def dxdt(I : np.array, E : np.array, A : np.array, K : float, alpha : float):
    # x: a column vector (N, 1)
    return -I + K * (A  @ np.tanh(alpha * E))

