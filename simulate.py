import numpy as np 
import networkx as nx
from network_gen import *
from opinion_dynamics import *
from censor import *
import matplotlib.pyplot as plt
import tqdm
import argparse 
import os

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1024, 
                    help='number of agents in the social network')
parser.add_argument('--m', type=int, default=10, )
parser.add_argument('--seed', type=int, default=42, help='random seed') 
parser.add_argument('--network-type', type=str, default='scale-free', 
                    help='network type (scale-free, complete, lfr-benchmark, small-world, communities, real-world)')
parser.add_argument('--beta', type=float, default=1, help='strength of homophily')
parser.add_argument('--enable-censorship', action='store_true', default=False, 
                    help='enable censorship')
parser.add_argument('--l', type=int, default=256, help='number of cliques')
parser.add_argument('--k', type=int, default=4, help='clique size')
parser.add_argument('--save-folder', type=str, default='./result', help='folder for saving plots')
parser.add_argument('--n-steps', type=int, default=1000, help='total timesteps') 

parser.add_argument('--K', type=float, default=3, help='strength of interaction')
parser.add_argument('--alpha', type=float, default=3, help='control the steepness of tanh')
parser.add_argument('--dt', type=float, default=0.01, help='timestep')

parser.add_argument('--tau1', type=float, default=3)
parser.add_argument('--tau2', type=float, default=1.5)
parser.add_argument('--mu', type=float, default=4)
parser.add_argument('--average-degree', type=float, default=4)
parser.add_argument('--min-community', type=int, default=20)

# Censorship related
parser.add_argument('--bias', type=float)
parser.add_argument('--favored', type=int)
parser.add_argument('--discount-factor', type=float)
parser.add_argument('--strength', type=float)
parser.add_argument('--T', type=float)

args = parser.parse_args()

np.random.seed(args.seed)


if args.network_type == 'scale-free':
    network = nx.barabasi_albert_graph(args.n, args.m, args.seed)
elif args.network_type == 'complete':
    network = nx.complete_graph(args.n)
elif args.network_type == 'lfr-benchmark':
    network = nx.LFR_benchmark_graph(args.n, args.tau1, args.tau2, args.mu, min_community=args.min_community,
                                     average_degree=args.average_degree, seed=args.seed)
elif args.network_type == 'small-world':
    network = nx.navigable_small_world_graph(args.n)
elif args.network_type == 'communities':
    network = nx.generators.community.connected_caveman_graph(args.l, args.k)
elif args.network_type == 'real-world':
    raise NotImplementedError 
else:
    raise ValueError

os.makedirs(args.save_folder, exist_ok=True)

society = Society(network, args.K, args.alpha, args.beta, args.dt)

if args.enable_censorship:
    censor = Censor(args.bias, args.favored, args.discount_factor, args.T, args.strength)
    society.couple(censor)

for i in tqdm.trange(args.n_steps):
    society.update()

plt.figure()
I, _ = society.trajectories()
for i in range(network.number_of_nodes()):
    plt.plot(I[i, :])
plt.savefig(os.path.join(args.save_folder, '%s_K=%.2f_alpha=%.2f_beta=%.2f.png' % (args.network_type, args.K, args.alpha, args.beta)))
