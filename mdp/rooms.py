import numpy as np


def layout_to_array(layout):
    return np.array([list(map(lambda c: 0 if c == 'w' else 1, line))
                     for line in layout.splitlines()])


def make_adjacency(layout):
    # UP, DOWN, LEFT, RIGHT
    directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]

    grid = layout_to_array(layout)
    state_to_grid_cell = np.argwhere(grid)
    grid_cell_to_state = {tuple(state_to_grid_cell[s].tolist()): s
                          for s in range(state_to_grid_cell.shape[0])}

    nstates = state_to_grid_cell.shape[0]
    nactions = len(directions)
    P = np.zeros((nactions, nstates, nstates))
    for state, idx in enumerate(state_to_grid_cell):
        for action, d in enumerate(directions):
            if grid[tuple(idx + d)]:
                dest_state = grid_cell_to_state[tuple(idx + d)]
                P[action, state, dest_state] = 1.
            else:
                P[action, state, state] = 1.  # self-loop if action cannot be taken

    return P, state_to_grid_cell


class FourRooms:
    def __init__(self):
        self.layout = """wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.P, self.state_to_grid_cell = make_adjacency(self.layout)
        self.R = np.copy(np.swapaxes(self.P[:, :, -1], 0, 1))
        self.P[:, -1, :] = 0.
        self.P[:, -1, -1] = 1.
        self.discount = 0.99
        self.mdp = [self.P, self.R, self.discount]
        self.nstates = self.P.shape[-1]
        self.nactions = self.P.shape[0]
