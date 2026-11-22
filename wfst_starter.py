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
    """Compose two FSTs with epsilon handling and tropical weights."""
    result = FST()

    start_pair = (a.start, b.start)
    state_map = {start_pair: result.add_state()}
    queue = [start_pair]

    while queue:
        sa, sb = queue.pop(0)
        src = state_map[(sa, sb)]

        if sa in a.finals and sb in b.finals:
            result.set_final(src, a.finals[sa] + b.finals[sb])

        # Epsilon output from A: advance A without advancing B.
        for il, ol, aw, da in a.arcs[sa]:
            if ol == EPS:
                pair = (da, sb)
                if pair not in state_map:
                    state_map[pair] = result.add_state()
                    queue.append(pair)
                result.add_arc(src, il, EPS, aw, state_map[pair])

        # Epsilon input to B: advance B without advancing A.
        for il, ol, bw, db in b.arcs[sb]:
            if il == EPS:
                pair = (sa, db)
                if pair not in state_map:
                    state_map[pair] = result.add_state()
                    queue.append(pair)
                result.add_arc(src, EPS, ol, bw, state_map[pair])

        # Match A output with B input.
        for ail, aol, aw, da in a.arcs[sa]:
            if aol == EPS:
                continue
            for bil, bol, bw, db in b.arcs[sb]:
                if bil == EPS or aol != bil:
                    continue
                pair = (da, db)
                if pair not in state_map:
                    state_map[pair] = result.add_state()
                    queue.append(pair)
                result.add_arc(src, ail, bol, aw + bw, state_map[pair])

    return result


if __name__ == "__main__":
    f = FST()
    s0, s1, s2 = f.add_state(), f.add_state(), f.add_state()
    f.add_arc(s0, "a", "x", 1, s1)
    f.add_arc(s1, "b", "y", 2, s2)
    f.add_arc(s0, "c", "z", 5, s2)
    f.set_final(s2)
    print(f)
    print(f.shortest_path())      # expect (3.0, ['a', 'b'], ['x', 'y'])
