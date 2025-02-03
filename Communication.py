import networkx as nx
import numpy as np

def insn_algorithm(sat_positions, max_isl_length, time_slices):
    ideal_topologies = []

    for t in time_slices:
        G_dense = nx.Graph()

        for sat in sat_positions[t]:
            G_dense.add_node(sat)

        for i, sat1 in enumerate(sat_positions[t]):
            for j, sat2 in enumerate(sat_positions[t]):
                if i < j:
                    distance = np.linalg.norm(np.array(sat1) - np.array(sat2))
                    if distance <= max_isl_length:
                        G_dense.add_edge(sat1, sat2, weight=distance)

        ideal_topology = nx.Graph()
        for sat1 in sat_positions[t]:
            for sat2 in sat_positions[t]:
                if sat1 != sat2:
                    path = nx.dijkstra_path(G_dense, sat1, sat2)
                    for k in range(len(path) - 1):
                        ideal_topology.add_edge(path[k], path[k+1])

        ideal_topologies.append(ideal_topology)

    return ideal_topologies

def dmnsn_algorithm(sat_positions, user_distribution, max_isl_length, time_slices):
    stable_topologies = []

    for t in time_slices:
        G_dense = nx.Graph()

        for sat in sat_positions[t]:
            G_dense.add_node(sat)

        for i, sat1 in enumerate(sat_positions[t]):
            for j, sat2 in enumerate(sat_positions[t]):
                if i < j:
                    distance = np.linalg.norm(np.array(sat1) - np.array(sat2))
                    if distance <= max_isl_length:
                        G_dense.add_edge(sat1, sat2, weight=distance)

        # Debugging: Print the nodes and edges
        print(f"Time slice {t}: Nodes in G_dense: {G_dense.nodes}")
        print(f"Time slice {t}: Edges in G_dense: {G_dense.edges}")

        stable_topology = nx.Graph()
        for user in user_distribution:
            src, dst = user
            if src in G_dense and dst in G_dense:
                path = nx.dijkstra_path(G_dense, src, dst)
                for k in range(len(path) - 1):
                    stable_topology.add_edge(path[k], path[k+1])
            else:
                print(f"Node {src} or {dst} not found in the graph for time slice {t}")

        stable_topologies.append(stable_topology)

    return stable_topologies

def mcsn_algorithm(ideal_topologies, stable_topologies):
    matching_degrees = []

    for ideal, stable in zip(ideal_topologies, stable_topologies):
        match_count = 0
        total_edges = len(ideal.edges)

        for edge in ideal.edges:
            if stable.has_edge(*edge):
                match_count += 1

        matching_degree = match_count / total_edges if total_edges > 0 else 0
        matching_degrees.append(matching_degree)

    return matching_degrees
def print_matching_degree_table(matching_degrees, max_isl_lengths):
    print("Max ISL Length | Matching Degree")
    print("---------------------------------")
    for length, degree in zip(max_isl_lengths, matching_degrees):
        print(f"{length:<15} | {degree:.2f}")

# Example data
sat_positions = {
    0: [(0, 0), (1, 1), (2, 2)],
    1: [(0, 0.1), (1, 1.1), (2, 2.1)],
}
user_distribution = [((0, 0), (2, 2)), ((1, 1), (0, 0))]
max_isl_length = 2.5
time_slices = [0, 1]

# Generate ideal topologies
ideal_topologies = insn_algorithm(sat_positions, max_isl_length, time_slices)

# Generate stable topologies
stable_topologies = dmnsn_algorithm(sat_positions, user_distribution, max_isl_length, time_slices)

# Evaluate matching degree
matching_degrees = mcsn_algorithm(ideal_topologies, stable_topologies)

# Example data for different max ISL lengths
max_isl_lengths = [2000, 3500, 5000]

# Assuming matching_degrees is calculated for each max ISL length
matching_degrees = [0.66, 0.44, 0.59]  # Example values

# Print the table
print_matching_degree_table(matching_degrees, max_isl_lengths)