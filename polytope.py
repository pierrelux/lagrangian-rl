import autograd.numpy as np


def dadashi_fig2d():
    """ Figure 2 d) of
    ''The Value Function Polytope in Reinforcement Learning''
    by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524

    Returns:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
    """
    P = np.array([[[0.7, 0.3], [0.2, 0.8]],
                  [[0.99, 0.01], [0.99, 0.01]]])
    R = np.array(([[-0.45, -0.1],
                   [0.5,  0.5]]))
    return P, R, 0.9


class DadashiFig2d:
    def __init__(self):
        self.mdp = dadashi_fig2d()
        self.P, self.R, self.discount = self.mdp
        self.nstates = self.P.shape[-1]
        self.nactions = self.P.shape[0]


if __name__ == "__main__":
    import graphviz as gv

    def mdp_to_dot(P, R, discount):
        del discount
        graph = gv.Digraph()
        graph.graph_attr['rankdir'] = 'LR'
        for a in range(P.shape[0]):
            for i in range(P.shape[1]):
                for j in range(P.shape[2]):
                    graph.edge(str(i), str(j), label=f"({a}, {R[i,a]}, {P[a,i,j]})")
        return graph.source

    dot_code = mdp_to_dot(*dadashi_fig2d())
    with open('mdp.dot', 'w') as file:
        file.write(dot_code)
