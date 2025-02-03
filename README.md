# Network Simulation Project
This project implements a comprehensive network simulation system that compares various multiple access techniques:
1. Dijkstra's algorithm for optimal path finding and explores Network Utility Maximization (NUM) using both primal and dual approaches
2. TDMA and optimization the flow and load
3. Non-Orthogonal Multiple Access (NOMA) and Orthogonal Multiple Access (OMA) simultaion. (Based on article : https://ieeexplore.ieee.org/document/7557079/)

## Main.py
- This file serves as the main interface for running simulations:
- Provides a menu-driven interface for selecting different simulation options.
- Implements functions for each simulation scenario

## Simulation.py
This file contains the core classes and functions for the network simulation:
Class:
- Vertex: Represents a node in the network.
- User: Represents a user in the network.
- Link: Represents a connection between vertices.
- Flow: Represents data flow in the network.
- Cluster: Represents a group of users for NOMA calculations.
- Network: The main class that represents the entire network and its operations.
Funcitons:
- Path calculation using Dijkstra's algorithm
- Network creation and visualization
- NOMA and OMA rate calculations
- Power allocation and interference calculations

## Functions.py
Utility functions and algorithms used throughout the simulation:
- Dijkstra's algorithm for shortest path calculation.
- Functions for TDMA rate setting and optimization.
- Visualization functions for flow rates and link utilization.
- Helper functions for NOMA cluster division and rate comparison. (Based on Article)
