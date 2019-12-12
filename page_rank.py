import random
import os
import time
from progress import Progress
import networkx as nx
import concurrent.futures

WEB_DATA = os.path.join(os.path.dirname(__file__), 'school_web.txt')


def load_graph(fd):
    """Load graph from text file

    Parameters:
    fd -- a file like object that contains lines of URL pairs

    Returns:
    A representation of the graph.

    Called for example with

    >>> graph = load_graph(open("web.txt"))

    the function parses the input file and returns a graph representation.
    Each line in the file contains two white space seperated URLs and
    denotes a directed edge (link) from the first URL to the second.
    """
    graph = nx.DiGraph()
    with fd as file:
    # Iterate through the file line by line
        for line in file:
            # And split each line into two URLs
            node, target = line.split()
            graph.add_edge(node,target)
    return graph


def print_stats(graph):
    """Print number of nodes and edges in the given graph"""
    print("Number of nodes: ", len(graph.nodes))
    print("Number of edges: ", len(graph.edges))


def stochastic_page_rank(graph, n_iter=1_000_000, n_steps=100):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of random walks performed
    n_steps (int) -- number of followed links before random walk is stopped

    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will after n_steps end
    on each node of the given graph.
    """
    start = time.time()
    outEdgy = {}
    for x in graph.nodes:
        for out in graph.out_edges(x):
            outEdgy.setdefault(out[0], []).append(out[1])
    print(outEdgy)
    par = Progress(n_iter, 'Calculating Stochastic PageRank:')
    hitcount = {}
    for node in graph.nodes():
        hitcount.setdefault(node, 0)
    for iter in range(n_iter):
        par += 1
        par.show()
        current_node = random.choice(list(graph.nodes))
        for steps in range(n_steps):
            current_node = random.choice(outEdgy[current_node])
        hitcount[current_node] += 1/n_iter
    stop = time.time()
    time_stochastic = stop - start
    return hitcount, time_stochastic

def distribution_page_rank(graph, n_iter=100):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of probability distribution updates

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """
    start = time.time()
    par = Progress(n_iter, 'Calculating Distribution PageRank: ')
    node_prob = {}
    next_prob = {}
    for nodes in graph.nodes:
        node_prob.setdefault(nodes, 1/len(graph.nodes))
    for times in range(n_iter):
        par += 1
        par.show()
        for set in graph.nodes:
            next_prob[set] = 0
        for node in graph.nodes:
            p = node_prob[node]/graph.out_degree(node)
            for target in graph.out_edges(node):
                next_prob[target[1]] += p
    node_prob = next_prob
    stop = time.time()
    time_probabilistic = stop - start
    return node_prob, time_probabilistic



def main():
    # Load the web structure from file
    web = load_graph(open(WEB_DATA))

    # print information about the website
    print_stats(web)

    # The graph diameter is the length of the longest shortest path
    # between any two nodes. The number of random steps of walkers
    # should be a small multiple of the graph diameter.
    diameter = 3
    with concurrent.futures.ProcessPoolExecutor() as pool:
        # Measure how long it takes to estimate PageRank through random walks
        print("Estimate PageRank through random walks:")
        n_iter = len(web)**2
        n_steps = 2*diameter
        p1 = pool.submit(stochastic_page_rank, web, n_iter, n_steps)
        p1r,time_stochastic  = p1.result()
        # Show top 20 pages with their page rank and time it took to compute
        top = sorted(p1r.items(), key=lambda item: item[1], reverse=True)
        print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:20]))
        print(f'Calculation took {time_stochastic:.2f} seconds.\n')

        # Measure how long it takes to estimate PageRank through probabilities
        print("Estimate PageRank through probability distributions:")
        n_iter = 2*100
        p2 = pool.submit(distribution_page_rank, web, n_iter)
        p2r, time_probabilistic = p2.result()
        # Show top 20 pages with their page rank and time it took to compute
        top = sorted(p2r.items(), key=lambda item: item[1], reverse=True)
        print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:20]))
        print(f'Calculation took {time_probabilistic:.2f} seconds.\n')

    # Compare the compute time of the two methods
    speedup = time_stochastic/time_probabilistic
    print(f'The probabilitic method was {speedup:.0f} times faster.')


if __name__ == '__main__':
    main()
