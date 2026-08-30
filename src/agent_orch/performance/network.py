from __future__ import annotations

import heapq
import math
from collections import defaultdict

from agent_orch.schema.models import LinkSpec


Edge = tuple[str, str]


class NetworkBackend:
    def __init__(self, links: tuple[LinkSpec, ...], slot_seconds: float = 1.0):
        self.links = {(link.source, link.target): link for link in links}
        self.slot_seconds = slot_seconds
        self.adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for link in links:
            self.adjacency[link.source].append((link.target, link.propagation_ms))
        self._path_cache: dict[tuple[str, str], tuple[Edge, ...]] = {}

    def path(self, source: str, target: str) -> tuple[Edge, ...]:
        if source == target:
            return ()
        key = (source, target)
        if key in self._path_cache:
            return self._path_cache[key]
        queue: list[tuple[float, str, tuple[Edge, ...]]] = [(0.0, source, ())]
        best = {source: 0.0}
        while queue:
            distance, node, path = heapq.heappop(queue)
            if node == target:
                self._path_cache[key] = path
                return path
            if distance > best.get(node, math.inf):
                continue
            for neighbor, weight in self.adjacency.get(node, []):
                candidate = distance + weight
                if candidate < best.get(neighbor, math.inf):
                    best[neighbor] = candidate
                    edge = (node, neighbor)
                    heapq.heappush(queue, (candidate, neighbor, path + (edge,)))
        raise ValueError(f"No physical path from {source} to {target}")

    def add_traffic(
        self,
        loads_mbit: dict[Edge, float],
        source: str,
        target: str,
        request_rate: float,
        data_mb_per_request: float,
    ) -> None:
        carried_mbit = request_rate * self.slot_seconds * data_mb_per_request * 8.0
        for edge in self.path(source, target):
            loads_mbit[edge] = loads_mbit.get(edge, 0.0) + carried_mbit

    def path_delay(
        self,
        source: str,
        target: str,
        loads_mbit: dict[Edge, float],
    ) -> float:
        delay = 0.0
        for edge in self.path(source, target):
            link = self.links[edge]
            delay += link.propagation_ms / 1000.0
            delay += loads_mbit.get(edge, 0.0) / link.capacity_mbps
        return delay

    def first_token_return_delay(
        self,
        source: str,
        target: str,
        token_data_mb: float,
    ) -> float:
        delay = 0.0
        for edge in self.path(source, target):
            link = self.links[edge]
            delay += link.propagation_ms / 1000.0
            delay += token_data_mb * 8.0 / link.capacity_mbps
        return delay

    def utilization(self, loads_mbit: dict[Edge, float]) -> dict[str, float]:
        return {
            f"{u}->{v}": loads_mbit.get((u, v), 0.0)
            / (link.capacity_mbps * self.slot_seconds)
            for (u, v), link in self.links.items()
        }

    def traffic_cost(self, loads_mbit: dict[Edge, float]) -> float:
        return sum(
            loads_mbit.get(edge, 0.0) / 8.0 * link.cost_per_mb
            for edge, link in self.links.items()
        )

