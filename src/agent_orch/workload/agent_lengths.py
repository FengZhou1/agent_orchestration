"""Synthetic message-set lengths with conditional sampling and hard budgets."""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist

import networkx as nx
import numpy as np

from .agent_graph import pattern_graph


@dataclass(frozen=True)
class LengthDistribution:
    lower: int
    upper: int
    mu: float
    sigma: float

    def draw(self, rng, minimum=None, maximum=None):
        lo = max(self.lower, int(math.ceil(self.lower if minimum is None else minimum)))
        hi = min(self.upper, int(math.floor(self.upper if maximum is None else maximum)))
        if lo > hi:
            raise ValueError("Required history exceeds the input length bound")
        if self.sigma <= 0 or self.lower <= 0 or self.lower > self.upper:
            raise ValueError("Invalid truncated lognormal")
        if lo == hi:
            return lo
        norm = NormalDist()
        a = norm.cdf((math.log(lo) - self.mu) / self.sigma)
        b = norm.cdf((math.log(hi) - self.mu) / self.sigma)
        q = min(1 - 1e-15, max(1e-15, a + float(rng.random()) * (b - a)))
        return min(hi, max(lo, round(math.exp(self.mu + self.sigma * norm.inv_cdf(q)))))


INPUT = LengthDistribution(256, 16384, 7.6246189861593985, 1.3099923490316212)
OUTPUT = LengthDistribution(16, 2048, 5.5112786586378215, 2.766327147128583)
TOOL = LengthDistribution(48, 5000, 5.0482551978514225, 1.7321669173732124)


def merge_messages(parents):
    merged = {}
    for messages in parents:
        for key, length in messages.items():
            if key in merged and merged[key] != length:
                raise ValueError("Shared message identity has inconsistent lengths")
            merged[key] = length
    return merged


def generate_lengths(graph, seed, input_distribution=INPUT, output_distribution=OUTPUT,
                     tool_distribution=TOOL, call_budget=32768, app_budget=128000,
                     max_attempts=200):
    # A message generated at one node may enter several later prefill requests.
    # The incidence matrix charges each such occurrence, not each graph path.
    llms = sorted(n for n, kind in graph.node_types.items() if kind == "llm")
    full = graph.graph()
    roots = [n for n in full if full.in_degree(n) == 0]
    if len(roots) != 1 or roots[0] not in llms:
        raise ValueError("Expected one entry LLM")
    position = {n: i for i, n in enumerate(llms)}
    rows, patterns, ancestors = [], [], {}
    for index, variant in enumerate(graph.patterns):
        pg = pattern_graph(variant)
        indices = []
        for node in nx.lexicographical_topological_sort(pg):
            ancestors[(index, node)] = nx.ancestors(pg, node)
            if graph.node_types[node] != "llm":
                continue
            coefficients = [int(n == node or n in ancestors[(index, node)]) for n in llms]
            indices.append(len(rows))
            rows.append((index, node, coefficients))
        patterns.append(indices)
    matrix = np.asarray([row[2] for row in rows], dtype=float)
    future_charges = np.asarray([matrix[indices].sum(axis=0) for indices in patterns])
    failures = {}
    for attempt in range(max_attempts):
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
        outputs = {n: output_distribution.draw(rng) for n in llms}
        tool_text = {n: tool_distribution.draw(rng) for n, kind in sorted(graph.node_types.items())
                     if kind == "tool"}
        row_outputs = np.asarray([outputs[node] for _, node, _ in rows])
        nonnew = np.asarray([sum(outputs.get(n, tool_text.get(n, 0))
                               for n in ancestors[(index, node)]) for index, node, _ in rows])
        upper = np.minimum(input_distribution.upper, call_budget - row_outputs)
        additions = np.zeros(len(llms))
        additions[position[roots[0]]] = input_distribution.lower
        try:
            inputs = nonnew + matrix @ additions
            totals = np.asarray([sum(inputs[indices] + row_outputs[indices]) for indices in patterns])
            if np.any(inputs > upper) or np.any(totals > app_budget):
                raise ValueError("Output/tool history leaves no feasible minimum input")
            full = graph.graph()
            for node in nx.lexicographical_topological_sort(full):
                if node not in position:
                    continue
                column = position[node]
                inputs = nonnew + matrix @ additions
                totals = np.asarray([sum(inputs[indices] + row_outputs[indices]) for indices in patterns])
                affected = matrix[:, column] > 0
                bound = float(np.min(upper[affected] - inputs[affected]))
                consuming = future_charges[:, column] > 0
                bound = min(bound, float(np.min(
                    (app_budget - totals[consuming]) / future_charges[consuming, column])))
                extra = max(0, int(math.floor(bound + 1e-8)))
                own_rows = [j for j, (_, n, _) in enumerate(rows) if n == node]
                weights = np.asarray([graph.patterns[rows[j][0]].probability for j in own_rows])
                history_mean = float(np.dot(
                    inputs[own_rows] - additions[column], weights) / weights.sum())
                current = int(additions[column])
                if extra:
                    target = input_distribution.draw(rng,
                        minimum=max(input_distribution.lower, math.ceil(history_mean + current)),
                        maximum=math.floor(history_mean + current + extra))
                    additions[column] = min(current + extra, max(current, round(target - history_mean)))
            inputs = nonnew + matrix @ additions
            details = {}
            for index, indices in enumerate(patterns):
                calls = {}
                for j in indices:
                    _, node, _ = rows[j]
                    messages = {}
                    for ancestor in sorted(ancestors[(index, node)]):
                        if ancestor in outputs:
                            if additions[position[ancestor]]:
                                messages[f"input:{ancestor}"] = int(additions[position[ancestor]])
                            messages[f"output:{ancestor}"] = outputs[ancestor]
                        else:
                            messages[f"tool:{ancestor}"] = tool_text[ancestor]
                    new = int(additions[position[node]])
                    history_tokens = sum(messages.values())
                    if new:
                        messages[f"input:{node}"] = new
                    prompt = sum(messages.values())
                    assert prompt == int(inputs[j])
                    if not input_distribution.lower <= prompt <= upper[j]:
                        raise ValueError("Jointly generated input exceeds a hard bound")
                    calls[node] = {"input_tokens": prompt, "output_tokens": outputs[node],
                                   "history_tokens": history_tokens, "new_input_tokens": new,
                                   "input_messages": dict(sorted(messages.items()))}
                total = sum(call["input_tokens"] + call["output_tokens"] for call in calls.values())
                if total > app_budget:
                    raise ValueError("Application token budget exceeded")
                details[f"p{index}"] = {"calls": calls, "total_tokens": total}
            return {"outputs": outputs, "tool_text": tool_text, "patterns": details,
                    "length_attempts": attempt + 1, "rejections": failures}
        except ValueError as error:
            key = str(error)
            failures[key] = failures.get(key, 0) + 1
    raise ValueError(f"No feasible joint lengths after {max_attempts} attempts: {failures}")


def representative_prompts(graph, lengths):
    numerator = {n: 0.0 for n, kind in graph.node_types.items() if kind == "llm"}
    mass = dict.fromkeys(numerator, 0.0)
    for index, variant in enumerate(graph.patterns):
        for node, call in lengths["patterns"][f"p{index}"]["calls"].items():
            numerator[node] += variant.probability * call["input_tokens"]
            mass[node] += variant.probability
    return {node: numerator[node] / mass[node] for node in numerator}


def weighted_quantile(values, weights, q):
    order = np.argsort(values, kind="stable")
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    if not len(v) or w.sum() <= 0:
        return None
    return float(v[min(len(v) - 1, np.searchsorted(np.cumsum(w), q * w.sum(), side="left"))])


def describe(values, weights=None):
    if not len(values):
        return {}
    v = np.asarray(values, dtype=float)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = float(np.dot(v, w))
    variance = float(np.dot((v - mean) ** 2, w))
    skew = float(np.dot((v - mean) ** 3, w) / variance ** 1.5) if variance > 0 else 0.0
    return {"min": float(v.min()), "max": float(v.max()), "mean": mean,
            "p50": weighted_quantile(v, w, .5), "p90": weighted_quantile(v, w, .9),
            "p99": weighted_quantile(v, w, .99), "skewness": skew}
