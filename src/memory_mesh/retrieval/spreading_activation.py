from __future__ import annotations

from typing import Dict, List, Set, Tuple
from memory_mesh.core.types import EntityNode
from memory_mesh.graph.store import KnowledgeGraphStore


class SpreadingActivationEngine:
    """
    Implements associative spreading activation across the semantic entity graph.
    Simulates human cognitive recall by activating related concepts through multi-hop relational pathways.
    """

    def __init__(
        self,
        decay_factor: float = 0.6,
        activation_threshold: float = 0.05,
        max_hops: int = 3,
    ):
        self.decay_factor = decay_factor
        self.activation_threshold = activation_threshold
        self.max_hops = max_hops

    def spread(
        self,
        graph: KnowledgeGraphStore,
        seed_nodes: List[EntityNode],
        initial_energy: float = 1.0,
    ) -> Tuple[Dict[str, float], List[List[str]]]:
        """
        Propagates energy from seed nodes outward along relation edges.
        Returns:
            - Dict of {node_id: activation_energy}
            - List of associative traversal paths
        """
        energies: Dict[str, float] = {}
        paths: List[List[str]] = []

        if not seed_nodes:
            return energies, paths

        # Initialize seeds with full energy
        current_frontier: Dict[str, float] = {}
        frontier_paths: Dict[str, List[str]] = {}

        for seed in seed_nodes:
            energies[seed.id] = initial_energy
            current_frontier[seed.id] = initial_energy
            frontier_paths[seed.id] = [seed.name]

        for hop in range(self.max_hops):
            next_frontier: Dict[str, float] = {}
            new_paths: Dict[str, List[str]] = {}

            for curr_node_id, curr_energy in current_frontier.items():
                if curr_energy < self.activation_threshold:
                    continue

                neighbors = graph.get_neighbors(curr_node_id, bidirectional=True)
                if not neighbors:
                    continue

                total_weight = sum(edge.weight for _, edge in neighbors) or 1.0
                curr_path = frontier_paths.get(curr_node_id, [curr_node_id])

                for neighbor_node, edge in neighbors:
                    # Do not bounce back to nodes already in this traversal path
                    if neighbor_node.name in curr_path:
                        continue

                    # Proportionate transfer decayed by distance
                    weight_ratio = edge.weight / total_weight
                    transferred = curr_energy * self.decay_factor * (0.6 + 0.4 * weight_ratio)

                    if transferred >= self.activation_threshold:
                        existing = energies.get(neighbor_node.id, 0.0)
                        # Diminishing returns on activation
                        new_val = round(existing + transferred * (1.0 - existing * 0.5), 4)
                        energies[neighbor_node.id] = new_val

                        next_frontier[neighbor_node.id] = max(
                            next_frontier.get(neighbor_node.id, 0.0), transferred
                        )

                        traversal = curr_path + [edge.relation, neighbor_node.name]
                        new_paths[neighbor_node.id] = traversal
                        paths.append(traversal)

            current_frontier = next_frontier
            frontier_paths.update(new_paths)

        return energies, paths
