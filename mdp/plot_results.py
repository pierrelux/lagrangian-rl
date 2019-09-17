import os
import pickle
import argparse
from itertools import product, cycle
from collections import defaultdict

import numpy as np
import tikzplotlib
import graphviz as gv
import matplotlib.pyplot as plt

import polytope


def policy_evaluation(P, R, gamma, policy):
    """ Policy Evaluation Solver

    We denote by 'A' the number of actions, 'S' for the number of
    states.

    Args:
      P (numpy.ndarray): Transition function as (A x S x S) tensor
      R (numpy.ndarray): Reward function as a (S x A) tensor
      gamma (float): Scalar discount factor
      policies (numpy.ndarray): tensor of shape (S x A)

    Returns:
      tuple (vf, qf) where the first element is vector of length S and the second element contains
      the Q functions as matrix of shape (S x A).
    """
    nstates = P.shape[-1]
    expanded = False
    if policy.ndim < 3:
        policy = np.expand_dims(policy, axis=0)
        expanded = True

    R = np.expand_dims(R, axis=0)
    ppi = np.einsum('ast,nsa->nst', P, policy)
    rpi = np.einsum('nsa,nsa->ns', R, policy)
    vf = np.linalg.solve(np.eye(nstates) - gamma*ppi, rpi)
    qf = R + gamma*np.einsum('ast,nt->nsa', P, vf)

    if expanded is True:
        vf = np.squeeze(vf)
        qf = np.squeeze(qf)

    return vf, qf


def make_polytope_figure(P, R, discount, rng, npolicies=100, box=None, delta=0.01, ):
    nstates, nactions = P.shape[-1], P.shape[0]
    random_policies = np.zeros((npolicies, nstates, nactions))
    random_policies[:, :, 0] = rng.uniform(size=(npolicies, nstates))
    random_policies[:, :, 1] = 1 - random_policies[:, :, 0]
    fig, ax = plt.subplots()

    vfs, _ = policy_evaluation(P, R, discount, random_policies)
    ax.scatter(vfs[:, 0], vfs[:, 1], s=12, alpha=1., zorder=0)
    state_action_cartesian_product = np.array(list(product(range(nactions), repeat=nstates)))

    if box is not None:
        def constraint(v0, v1, s=0, a=0):
            val = discount*P[a, s, 0]*v0 + discount*P[a, s, 1]*v1
            if s == 0:
                return v0 - val
            return v1 - val

        box = np.asarray(box)
        if box.ndim == 0:
            s0_valrange = np.arange(-box, box, delta)
            s1_valrange = s0_valrange
        else:
            s0_valrange = np.arange(box[0, 0], box[0, 1], delta)
            s1_valrange = np.arange(box[1, 0], box[1, 1], delta)

        vstate0, vstate1 = np.meshgrid(s0_valrange, s1_valrange)

        for state, action in state_action_cartesian_product:
            cp = ax.contour(vstate0, vstate1, constraint(vstate0, vstate1, state, action),
                            levels=[R[state, action]], zorder=5)
            ax.clabel(cp, fmt=f"$r(s_{state}, a_{action})$", inline=1)  # , fontsize=8)

    deterministic_policies = np.eye(nactions)[state_action_cartesian_product]
    dvfs, _ = policy_evaluation(P, R, discount, deterministic_policies)
    ax.scatter(dvfs[:, 0], dvfs[:, 1], c='r', zorder=10)

    return fig, ax


def mdp_to_dot(P, R, discount, bend_delta=10, draw_initial=False):
    del discount
    graph = gv.Digraph(
        body=['d2tdocpreamble = "\\usetikzlibrary{automata}"',
              'd2tfigpreamble = "\\tikzstyle{every state}= [draw=blue!50,semithick,fill=blue!20]"'],
        node_attr={'style': 'state'},
        edge_attr={'lblstyle': 'auto'})
    graph.graph_attr['rankdir'] = 'LR'

    nstates, nactions = P.shape[-1], P.shape[0]

    if draw_initial is True:
        for i in range(nstates):
            graph.node(str(i), style="state, initial")

    edge_bends = defaultdict(lambda: bend_delta)
    edge_loops = defaultdict(lambda: cycle(['above', 'below']))
    for a, i, j in product(range(nactions), range(nstates), range(nstates)):
        if P[a, i, j] > 1e-5:
            edge_spec = {'tail_name': str(i), 'head_name': str(
                j), 'label': f"({a}, {R[i,a]:.3f}, {P[a,i,j]:.3g})"}
            if i == j:
                edge_spec['topath'] = "loop {}".format(next(edge_loops[i]))
            else:
                key = (min(i, j), max(i, j))
                edge_spec['topath'] = "bend left={:d}".format(edge_bends[key])
                edge_bends[key] += bend_delta
            graph.edge(**edge_spec)

    return graph.source


def dot_to_tex(dotfilename):
    os.system(f"dot2tex -ftikz --tikzedgelabel -c {dotfilename}.dot > {dotfilename}.tex")


def tex_to_pdf(texfilename):
    os.system(f"pdflatex {texfilename}.tex")


def export_graph(prefix, dot_code):
    with open(f"{prefix}.dot", 'w') as fp:
        fp.write(dot_code)
    dot_to_tex(prefix)
    tex_to_pdf(prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('solution', type=str)
    arguments = parser.parse_args()

    with open(arguments.solution, 'rb') as fp:
        synthetic_transition, synthetic_reward = pickle.load(fp)

    mdp = polytope.DadashiFig2d()
    rng = np.random.RandomState(0)

    make_polytope_figure(synthetic_transition, synthetic_reward, mdp.discount,
                         rng, npolicies=500)  # box=[[-0.25, 0.25], [-0.1, 0.1]])
    plt.savefig('synthetic_polytope.pdf')
    tikzplotlib.save('synthetic_polytope.tex')

    make_polytope_figure(*mdp.mdp, rng, npolicies=500)  # box=[[-2.25, 1], [-2.25, 2.25]])
    plt.savefig('true_polytope.pdf')
    tikzplotlib.save('true_polytope.tex')

    source = mdp_to_dot(synthetic_transition, synthetic_reward, mdp.discount)
    export_graph('synthetic', source)
    export_graph('true', mdp_to_dot(*mdp.mdp))
