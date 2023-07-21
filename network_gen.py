import networkx as nx

def generate_network(network_type, **args):
    if network_type == 'scale-free':
        return nx.barabasi_albert_graph(**args)
    elif network_type == 'complete':
        return nx.complete_graph(**args)
    elif network_type == 'lfr-benchmark':
        return nx.LFR_benchmark_graph(**args)
    elif network_type == 'small-world':
        return nx.navigable_small_world_graph(**args)
    elif network_type == 'communities':
        return nx.generators.community.connected_caveman_graph(**args)
    elif network_type == 'real-world':
        raise NotImplementedError 
    else:
        raise ValueError