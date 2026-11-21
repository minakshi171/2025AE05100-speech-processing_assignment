"""
wfst_starter.py  --  starter skeleton for Part 5 (tropical semiring: weights add along a path,
the best path is the one with the smallest total weight).

Given: the FST container, arc handling and shortest path for acyclic machines.
Yours to write [Critical]: compose(a, b).
"""
EPS = "<eps>"


class FST:
    def __init__(self):
        self.arcs = {}          # state -> list of (ilabel, olabel, weight, next_state)
        self.start = 0
        self.finals = {}        # state -> final weight
        self.n_states = 0

    def add_state(self):
        s = self.n_states
        self.n_states += 1
        self.arcs[s] = []
        return s

    def add_arc(self, src, ilabel, olabel, weight, dst):
        self.arcs[src].append((ilabel, olabel, float(weight), dst))

    def set_final(self, state, weight=0.0):
        self.finals[state] = float(weight)

    def shortest_path(self):
        """Best (weight, input labels, output labels) from start to a final state. Acyclic only."""
        order, seen = [], set()

        def visit(u):
            if u in seen:
                return
            seen.add(u)
            for (_, _, _, v) in self.arcs[u]:
                visit(v)
            order.append(u)
        visit(self.start)
        order.reverse()
        best = {self.start: (0.0, [], [])}
        for u in order:
            if u not in best:
                continue
            w, ins, outs = best[u]
            for (il, ol, aw, v) in self.arcs[u]:
                cand = (w + aw, ins + ([il] if il != EPS else []), outs + ([ol] if ol != EPS else []))
                if v not in best or cand[0] < best[v][0]:
                    best[v] = cand
        done = [(best[s][0] + fw,) + best[s][1:] for s, fw in self.finals.items() if s in best]
        return min(done, key=lambda t: t[0]) if done else None

    def __str__(self):
        lines = []
        for s in range(self.n_states):
            for (il, ol, w, d) in self.arcs[s]:
                lines.append(f"{s} -> {d}  {il}:{ol}/{w}")
        lines.append("finals: " + str(self.finals))
        return "\n".join(lines)


def compose(a, b):
    """TODO [Critical]: return the composition of a and b (tropical semiring).
    A path in the result exists for every pair of paths where a's output label equals b's input
    label. Handle epsilon labels. Weights add."""
    raise NotImplementedError


if __name__ == "__main__":
    f = FST()
    s0, s1, s2 = f.add_state(), f.add_state(), f.add_state()
    f.add_arc(s0, "a", "x", 1, s1)
    f.add_arc(s1, "b", "y", 2, s2)
    f.add_arc(s0, "c", "z", 5, s2)
    f.set_final(s2)
    print(f)
    print(f.shortest_path())      # expect (3.0, ['a', 'b'], ['x', 'y'])
