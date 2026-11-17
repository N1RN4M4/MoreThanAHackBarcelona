"""Turn per-frame evidence into whole possessions: a Viterbi decode over H, A and D.

A possession is an interval, and spellF1 scores it as one, so the decision is made over the sequence
rather than frame by frame. Evidence accumulates inside a stretch, and a change of state has to earn
its penalty. Unlike the baseline's centred vote this does not blur boundaries: a change lands where
the evidence turns, instead of half a window from it.
"""

import numpy as np

# Params
H, A, D = 0, 1, 2
N_STATES = 3


def transition_costs(switch, to_dead, from_dead):
    """3x3 cost of moving from row state to column state."""
    costs = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    costs[H, A] = costs[A, H] = switch
    costs[H, D] = costs[A, D] = to_dead
    costs[D, H] = costs[D, A] = from_dead
    return costs


def viterbi(emissions, costs):
    """Best state per frame for one half. emissions is (n, 3) log-evidence; float64 and exact.

    On a tie the path that changes state later wins. CHALLENGE.md: possession stays with the last
    team that had it, so among equally good readings the change is at the new team's first evidence
    and not at the old team's last.
    """
    n = len(emissions)
    if not n:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, N_STATES), dtype=np.int8)
    score = emissions[0].copy()
    # Plain Python floats: a three-state recursion is far quicker this way than through numpy.
    table = emissions.tolist()
    c = costs.tolist()
    s0, s1, s2 = float(score[0]), float(score[1]), float(score[2])
    c01, c02, c10, c12, c20, c21 = c[0][1], c[0][2], c[1][0], c[1][2], c[2][0], c[2][1]
    pointers = [None] * n
    for t in range(1, n):
        e0, e1, e2 = table[t]

        best0, from0 = s0, 0
        v = s1 - c10
        if v >= best0:
            best0, from0 = v, 1
        v = s2 - c20
        if v >= best0:
            best0, from0 = v, 2

        best1, from1 = s1, 1
        v = s0 - c01
        if v >= best1:
            best1, from1 = v, 0
        v = s2 - c21
        if v >= best1:
            best1, from1 = v, 2

        best2, from2 = s2, 2
        v = s0 - c02
        if v >= best2:
            best2, from2 = v, 0
        v = s1 - c12
        if v >= best2:
            best2, from2 = v, 1

        s0, s1, s2 = best0 + e0, best1 + e1, best2 + e2
        pointers[t] = (from0, from1, from2)

    state = 0
    if s1 > s0:
        state = 1
    if s2 > max(s0, s1):
        state = 2
    path = np.zeros(n, dtype=np.int8)
    path[n - 1] = state
    for t in range(n - 1, 0, -1):
        state = pointers[t][state]
        path[t - 1] = state
    return path
