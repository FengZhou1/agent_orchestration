"""Budgeted structured expansion; no deployment, policy or LLM invocation."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Variant:
    nodes: frozenset[str]
    edges: frozenset[tuple[str, str]]
    entries: tuple[str, ...]
    exits: tuple[str, ...]
    probability: float = 1.0


@dataclass
class Fragment:
    variants: list[Variant]
    control: dict


def compose(children: list[Fragment], mode: str, weights=None) -> Fragment:
    """OR adds variants; SEQ/AND take their complete Cartesian product."""
    if not children:
        return Fragment([Variant(frozenset(), frozenset(), (), ())], {"op": "empty"})
    if mode == "exclusive":
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(children) or np.any(weights <= 0):
            raise ValueError("Exclusive alternatives require strictly positive weights")
        weights = weights / weights.sum()
        variants = [Variant(v.nodes, v.edges, v.entries, v.exits,
                            v.probability * float(w))
                    for child, w in zip(children, weights) for v in child.variants]
    else:
        variants = []
        for selected in product(*(child.variants for child in children)):
            nonempty = [v for v in selected if v.nodes]
            nodes = frozenset().union(*(v.nodes for v in selected))
            edges = frozenset().union(*(v.edges for v in selected))
            if mode == "serial":
                edges |= frozenset((u, w) for left, right in zip(nonempty, nonempty[1:])
                                   for u in left.exits for w in right.entries)
                entries, exits = (nonempty[0].entries, nonempty[-1].exits) if nonempty else ((), ())
            elif mode == "parallel":
                entries = tuple(u for v in nonempty for u in v.entries)
                exits = tuple(u for v in nonempty for u in v.exits)
            else:
                raise ValueError(f"Unknown composition {mode}")
            variants.append(Variant(nodes, edges, entries, exits,
                                    math.prod(v.probability for v in selected)))
    control = {"op": mode, "children": [c.control for c in children]}
    if mode == "exclusive":
        control["weights"] = weights.tolist()
    return Fragment(variants, control)


@dataclass
class ExpandedGraph:
    node_types: dict[str, str]
    patterns: list[Variant]
    control: dict
    operation_counts: dict[str, int]

    def graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from((node, {"type": kind}) for node, kind in self.node_types.items())
        g.add_edges_from(edge for v in self.patterns for edge in v.edges)
        return g


class GraphExpander:
    def __init__(self, rng, max_patterns=32, fanout=4, operation_weights=None,
                 root_stop=(.05, .30), short_stop=(.60, .85), later_stop=(.15, .55)):
        self.rng = rng
        self.max_patterns = max_patterns
        self.fanout = fanout
        self.operation_weights = operation_weights or dict.fromkeys(("serial", "parallel", "exclusive", "repeat"), 1.)
        self.root_stop, self.short_stop, self.later_stop = root_stop, short_stop, later_stop
        self.types = {}
        self.counts = {}
        self.serials = {"llm": 0, "tool": 0}

    def atom(self, kind):
        node = ("l" if kind == "llm" else "t") + str(self.serials[kind])
        self.serials[kind] += 1
        self.types[node] = kind
        return Fragment([Variant(frozenset([node]), frozenset(), (node,), (node,))],
                        {"op": kind, "node": node})

    def record(self, mode):
        self.counts[mode] = self.counts.get(mode, 0) + 1

    def choose_mode(self, modes):
        weights = np.asarray([self.operation_weights[m] for m in modes], dtype=float)
        if weights.sum() <= 0:
            raise ValueError("No positive expansion weight fits the current budget")
        return str(self.rng.choice(modes, p=weights / weights.sum()))

    def split(self, total, width, minimum=0):
        rest = total - width * minimum
        if rest < 0:
            raise ValueError("Insufficient node budget")
        return (self.rng.multinomial(rest, np.ones(width) / width) + minimum).tolist()

    def tools(self, count, cap):
        if not count:
            return compose([], "serial")
        if count == 1:
            return self.atom("tool")
        modes = ["serial", "parallel"] + (["exclusive"] if cap >= 2 else [])
        mode = self.choose_mode(modes)
        width = int(self.rng.integers(2, min(self.fanout, count, cap if mode == "exclusive" else count) + 1))
        allocation = self.split(count, width, 1)
        # Reserve a constructive product/sum bound BEFORE creating branches.
        child_cap = max(1, cap // width) if mode == "exclusive" else max(1, int(cap ** (1 / width)))
        children = [self.tools(n, child_cap) for n in allocation]
        weights = self.rng.dirichlet(np.ones(width)) if mode == "exclusive" else None
        fragment = compose(children, mode, weights)
        if len(fragment.variants) > cap:
            raise AssertionError("Constructive pattern budget failed")
        self.record(mode + "_tools")
        return fragment

    def loop(self, llms, tools, cap):
        """A finite fresh-node expansion, possibly branching into reasoning paths."""
        if llms < 1:
            raise ValueError("A continuation must end in an LLM")
        modes = ["serial", "repeat"]
        if llms >= 3:
            modes += ["parallel"]
            if cap >= 2:
                modes += ["exclusive"]
        mode = self.choose_mode(modes)
        if mode in ("serial", "repeat"):
            parts = self.split(tools, llms)
            def extend(position, available):
                count_left = len(parts) - position
                can_respond = count_left > 1 and available >= 2
                prefix_cap = min(2, available // 2) if can_respond else 1
                prefix = compose([self.tools(parts[position], max(1, prefix_cap)),
                                  self.atom("llm")], "serial")
                if count_left == 1:
                    return prefix
                continuation_cap = max(1, (available - len(prefix.variants)
                    if can_respond else available) // len(prefix.variants))
                rest = extend(position + 1, continuation_cap)
                continued = compose([prefix, rest], "serial")
                if not can_respond:
                    return continued
                stop = float(self.rng.uniform(*self.later_stop))
                self.record("direct_response")
                return compose([prefix, continued], "exclusive", [stop, 1 - stop])
            result = extend(0, cap)
            self.record("finite_repeat" if mode == "repeat" else "continue_reasoning")
        else:
            width = int(self.rng.integers(2, min(self.fanout, llms - 1,
                                                cap if mode == "exclusive" else llms) + 1))
            lparts = self.split(llms - 1, width, 1)
            tparts = self.split(tools, width)
            child_cap = max(1, cap // width) if mode == "exclusive" else max(1, int(cap ** (1 / width)))
            branches = [self.loop(l, t, child_cap) for l, t in zip(lparts, tparts)]
            weights = self.rng.dirichlet(np.ones(width)) if mode == "exclusive" else None
            result = compose([compose(branches, mode, weights), self.atom("llm")], "serial")
            self.record(mode + "_reasoning")
        if len(result.variants) > cap:
            raise AssertionError("Continuation exceeded its constructive pattern budget")
        return result

    def expand(self, llms, tools):
        root = self.atom("llm")
        if llms == 1:
            if tools:
                raise ValueError("A one-LLM graph cannot initiate and summarize tools")
            return ExpandedGraph(self.types, root.variants, root.control, {"pure_inference": 1})
        if tools < 1:
            raise ValueError("Non-pure applications require tools")
        remaining = llms - 2
        first_tools = tools if not remaining else int(self.rng.integers(0, min(tools, self.fanout) + 1))
        # Reasoning may precede the first tool. All allocated tools still occur
        # in positive continuations; direct response is a control choice.
        first = compose([root, self.tools(first_tools, 2), self.atom("llm")], "serial")
        p_direct = float(self.rng.uniform(*self.root_stop))
        short_stop = 1.0 if not remaining else float(self.rng.uniform(*self.short_stop))
        patterns = [Variant(v.nodes, v.edges, v.entries, v.exits, p_direct)
                    for v in root.variants]
        patterns += [Variant(v.nodes, v.edges, v.entries, v.exits,
                             (1 - p_direct) * short_stop * v.probability)
                     for v in first.variants]
        control = {"op": "response_or_extend", "direct_probability": p_direct,
                   "short_stop_probability": short_stop, "first": first.control}
        self.record("direct_response")
        if remaining:
            cap = (self.max_patterns - len(patterns)) // len(first.variants)
            if cap < 1:
                raise ValueError("Pattern cap too small for the root response choices")
            tail = self.loop(remaining, tools - first_tools, cap)
            full = compose([first, tail], "serial")
            patterns += [Variant(v.nodes, v.edges, v.entries, v.exits,
                                 (1 - p_direct) * (1 - short_stop) * v.probability)
                         for v in full.variants]
            control["tail"] = tail.control
        if len(patterns) > self.max_patterns:
            raise AssertionError("Patterns must never be truncated after expansion")
        return ExpandedGraph(self.types, patterns, control, self.counts)


def pattern_graph(variant):
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(variant.nodes))
    graph.add_edges_from(sorted(variant.edges))
    return graph


def replay_control(control, node_types, cap=32):
    """Independently reconstruct the complete pattern space from control operators."""
    op = control["op"]
    if op in ("llm", "tool"):
        node = control["node"]
        if node_types[node] != op:
            raise ValueError("Control atom and node type disagree")
        return Fragment([Variant(frozenset([node]), frozenset(), (node,), (node,))], control)
    if op == "empty":
        return compose([], "serial")
    if op == "response_or_extend":
        first = replay_control(control["first"], node_types, cap)
        root_id = first.variants[0].entries[0]
        root = Fragment([Variant(frozenset([root_id]), frozenset(), (root_id,), (root_id,))], {})
        p, stop = control["direct_probability"], control["short_stop_probability"]
        if "tail" in control:
            tail = replay_control(control["tail"], node_types, cap)
            if 1 + len(first.variants) + len(first.variants) * len(tail.variants) > cap:
                raise ValueError("Control pattern count exceeds its pre-expansion limit")
            continued = compose([first, tail], "serial")
            return compose([root, first, continued], "exclusive", [p, (1-p)*stop, (1-p)*(1-stop)])
        return compose([root, first], "exclusive", [p, 1-p])
    children = [replay_control(child, node_types, cap) for child in control["children"]]
    count = sum(len(c.variants) for c in children) if op == "exclusive" else math.prod(len(c.variants) for c in children)
    if count > cap:
        raise ValueError("Control pattern count exceeds its pre-expansion limit")
    return compose(children, op, control.get("weights"))


def validate_graph(graph: ExpandedGraph, max_depth=30, max_llms=15, max_tools=30):
    full = graph.graph()
    if not nx.is_directed_acyclic_graph(full):
        raise ValueError("Full application graph is cyclic")
    roots = tuple(n for n in full if full.in_degree(n) == 0)
    if len(roots) != 1 or graph.node_types[roots[0]] != "llm":
        raise ValueError("An Agent application needs one entry LLM")
    lcount = sum(t == "llm" for t in graph.node_types.values())
    tcount = sum(t == "tool" for t in graph.node_types.values())
    pure = lcount == 1 and tcount == 0
    depth = nx.dag_longest_path_length(full) + 1
    if not (1 <= lcount <= max_llms and 0 <= tcount <= max_tools
            and (pure or (lcount >= 2 and tcount >= 1))
            and (depth == 1 if pure else 2 <= depth <= max_depth)):
        raise ValueError("Graph violates the joint structural bounds")
    if abs(sum(p.probability for p in graph.patterns) - 1) > 1e-10:
        raise ValueError("Pattern probabilities do not sum to one")
    covered = set()
    for variant in graph.patterns:
        pg = pattern_graph(variant)
        if variant.entries != roots:
            raise ValueError("A pattern must preserve the application entry")
        if variant.probability <= 0 or len(variant.exits) != 1:
            raise ValueError("Each positive pattern needs one final response")
        final = variant.exits[0]
        if graph.node_types[final] != "llm" or not nx.is_directed_acyclic_graph(pg):
            raise ValueError("Invalid pattern final node or dependence")
        if any(not nx.has_path(pg, n, final) for n in pg):
            raise ValueError("A pattern contains a dangling dependency")
        covered.update(pg)
    if covered != set(full):
        raise ValueError("Every full-graph node must have positive execution probability")
    return {"llm_nodes": lcount, "tool_nodes": tcount, "depth": depth,
            "width": max(len(layer) for layer in nx.topological_generations(full)),
            "patterns": len(graph.patterns), "pure": pure}
