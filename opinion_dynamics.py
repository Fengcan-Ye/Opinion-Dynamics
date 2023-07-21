import networkx as nx
import numpy as np

import copy

def initialize_opinions(network : nx.Graph):
    for i in range(network.number_of_nodes()):
        network.nodes[i]['intrinsic opinion'] = [np.random.uniform(-1, 1)]
        network.nodes[i]['extrinsic opinion'] = copy.copy(network.nodes[i]['intrinsic opinion'])

def current_opinions(network : nx.Graph, key = 'intrinsic opinion'):
    return np.array([network.nodes[i][key][-1] for i in range(network.number_of_nodes())])

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


def update_opinions(network : nx.Graph, 
                    K : float, 
                    alpha : float, 
                    beta : float,
                    discount_factor : float,
                    dt : float):
    intrinsics = current_opinions(network, 'intrinsic opinion')
    extrinsics = current_opinions(network, 'extrinsic opinion')

    # P(i is influenced by j) ~ |I_i - E_j|^{-beta}  where j is a neighbor of i
    p_unnormalized = np.abs(intrinsics.reshape(-1, 1) - extrinsics) ** (-beta) * nx.adjacency_matrix(network)
    p_unnormalized = np.nan_to_num(p_unnormalized, nan=0, posinf=0, neginf=0)
    p = p_unnormalized / np.sum(p_unnormalized, axis=1).reshape(-1, 1)

    us = np.random.uniform(0, 1, size = p.shape)
    A = p > us
    
    intrinsics = intrinsics + RK4(intrinsics, lambda I: dxdt(I, extrinsics, A, K, alpha), dt)

    for i in range(network.number_of_nodes()):
        network.nodes[i]['intrinsic opinion'].append(intrinsics[i])
        network.nodes[i]['extrinsic opinion'].append(intrinsics[i])


