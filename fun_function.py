import random
import heapq
import matplotlib.pyplot as plt
import copy
from Fun import Cluster
import numpy as np
def CalcNetworkRate(network, alpha, Algorithm, N=1e5):
    network.initial_users_rates()
    algorithm_functions = {"Primal": CalcPrimalRate, "Dual": CalcDualRate}
    CalcRate = algorithm_functions.get(Algorithm)
    users = network.users

    xAxis = []
    yAxis = []
    for _ in users:  # initialize the graph
        xAxis.append([])
        yAxis.append([])

    for i in range(int(N)):
        curUser = random.choice(users)
        id = curUser.Uid-1
        x_r = curUser.rate
        curUser.rate = CalcRate(curUser, users, alpha, x_r)
        xAxis[id].append(i)
        yAxis[id].append(curUser.rate)

    PrintRateResults(xAxis, yAxis, users, alpha, Algorithm)


def CalcPrimalRate(user, users, alpha, x_r, stepSize=0.0001):
    if alpha == float("inf"):
        avg_rate = sum(u.rate for u in users) / len(users)
        return max(0, min(1, avg_rate + stepSize)) if user.rate < avg_rate else max(0, user.rate - stepSize)

    payment = 0
    for link in user.links:  # calculate the payment of the user
        rateSum = 0
        for u in users:  # calculate the sum of the rates of all the users on the link
            if link in u.links:
                rateSum += u.rate
        payment += penaltyFunction(rateSum, link.total_capacity)
    return stepSize * (pow(user.rate, -1 * alpha) - payment) + x_r  # calculate the next rate of the user


def penaltyFunction(rate, capacity):
    if rate < capacity:
        return rate * capacity
    else:
        try:
            return pow(rate, 3) * 2
        except OverflowError: # TODO: check why it is overflow error
            return 0


def CalcDualRate(user, users, alpha, x_r, stepSize=0.0001):
    """ this function calculates the next rate of a given user for the dual algorithm """

    if alpha == float("inf"):
        # Adjusting based on the max constraint violation
        max_excess = max((sum(u.rate for u in users if link in u.links) - link.total_capacity) for link in user.links)
        return max(0, min(1, x_r - stepSize * max_excess))

    Q_l = 0
    for link in user.links:  # calculate the payment of the user
        rateSum = sum(u.rate for u in users if link in u.links) # Y_l
        L_delta = (rateSum - link.total_capacity) * stepSize
        link.LagrangianMultiplier = max(0, link.LagrangianMultiplier + L_delta)
        Q_l += link.LagrangianMultiplier
    if Q_l == 0:
        print("Ql is zero!")
    return pow(Q_l, -1/alpha) if Q_l != 0 else 0 # the inverse function of the utilization function


def PrintRateResults(xAxis, yAxis, users, alpha, Algorithm):
    plt.figure()
    print(f"-------------------------")
    print(f"{Algorithm} Algorithm, alpha={str(alpha)} Results:")
    sum_rate = 0
    for user in users:
        sum_rate += user.rate
        print(f"user {user.Uid} rate : {round(user.rate,2)}")
    print(f"sum_rate={round(sum_rate,2)}\n")
    for i in range(len(xAxis)):
        plt.plot(xAxis[i], yAxis[i], label=f"user {i+1}")
    plt.title(f"{Algorithm} Algorithm, alpha={str(alpha)}")
    plt.xlabel("Iteration Number")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid()
    plt.show(block=False)


def dijkstra_algorithm(network, start_vertex):
    '''
    preforming dijkstra for a speficiv staring vertex.
    if we have a vertrex that improve our distance we add it to the min heap
    preforming the algo until no new distnace improvment
    
    '''
    # Initialize distances to all vertices in the network as infinity, except for the start vertex set to 0
    distances = {vertex: float('inf') for vertex in network.vertices}
    previous_nodes = {vertex: None for vertex in network.vertices}
    distances[start_vertex] = 0
    priority_queue = [(0, start_vertex)]  # Priority queue to manage the exploration of vertices

    # The main loop continues until there are no more vertices to explore
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)  # Pop the vertex with the lowest distance
        if current_distance > distances[current_vertex]:  # If the popped distance is greater then known stop explore
            continue

        for neighbor, link in current_vertex.neighbors.items(): # Explore each neighbor of the current vertex
            distance = current_distance + link.weight

            if distance < distances[neighbor]:  # If the new distance is less than the previous update route
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_vertex
                heapq.heappush(priority_queue, (distance, neighbor))  # Push the updated distance for further explore

    return distances, previous_nodes


def set_flows_rate_based_on_tdma(network, K):

    '''
    for each link we iterate over the user above that link.
    we calcualte the flow above this link, and sum and update with capapcity and data
    
    '''
    for link in network.links:
        flows = []
        total_data = [0] * K
        users_using_link = [user for user in network.users if link in user.links]
        for user in users_using_link:
            for flow in user.flows:
                flows.append(flow)
                total_data[flow.channel] += flow.data_amount

        for flow in flows:
            flow.rate_by_links[link] = link.channels_capacities[flow.channel] * flow.data_amount / total_data[flow.channel]

    for flow in network.flows:
        flow.set_rate_2_min_of_rate_by_links()

    visualize_flow_rates_and_link_utilization(network, K)


def optimize_flows_rate_based_on_tdma(network, K):
    '''
    attempts to balance the data distribution across channels by moving flows from 
    the most loaded channel to the least loaded one
    '''
    for link in network.links:
        flows = []
        channel_flows = {k: [] for k in range(K)}
        total_data_in_channel = [0] * K
        users_using_link = [user for user in network.users if link in user.links]
        for user in users_using_link:
            for flow in user.flows:
                flows.append(flow)
                channel_flows[flow.channel].append(flow)
                total_data_in_channel[flow.channel] += flow.data_amount

        if all(not flows for flows in channel_flows.values()):  # check if channel_flows
            continue

        while True:
            max_data, max_channel = max((value, index) for index, value in enumerate(total_data_in_channel))
            min_data, min_channel = min((value, index) for index, value in enumerate(total_data_in_channel))
            diff = max_data - min_data

            temp_total_data_in_channel = copy.copy(total_data_in_channel)
            temp_channel_flows = copy.deepcopy(channel_flows)

            min_data_flow_from_max_channel = min(temp_channel_flows[max_channel], key=lambda flow: flow.data_amount)
            temp_channel_flows[max_channel].remove(min_data_flow_from_max_channel)
            temp_total_data_in_channel[max_channel] -= min_data_flow_from_max_channel.data_amount
            temp_channel_flows[min_channel].append(min_data_flow_from_max_channel)
            temp_total_data_in_channel[min_channel] += min_data_flow_from_max_channel.data_amount

            max_data, max_channel = max((value, index) for index, value in enumerate(temp_total_data_in_channel))
            min_data, min_channel = min((value, index) for index, value in enumerate(temp_total_data_in_channel))
            new_diff = max_data - min_data
            if new_diff >= diff:
                break  # Break the loop if no improvement in fairness diff

            total_data_in_channel = temp_total_data_in_channel
            channel_flows = temp_channel_flows

        total_data_in_channel = [max(1, td) for td in total_data_in_channel]  # Ensure min data is one to guaranty rate
        total_data_in_link = sum(total_data_in_channel)
        link.channels_capacities = [link.total_capacity * (td / total_data_in_link) for td in total_data_in_channel]

        for flow in flows:
            flow.rate_by_links[link] = link.channels_capacities[flow.channel] * flow.data_amount / total_data_in_channel[flow.channel]

    for flow in network.flows:
        flow.set_rate_2_min_of_rate_by_links()

    visualize_flow_rates_and_link_utilization(network, K)

# Orya: change below
def visualize_flow_rates_and_link_utilization(network, K):
    # Data for Flow Rates vs. Data Amounts
    data_amounts = [flow.data_amount for flow in network.flows]
    flow_rates = [flow.rate for flow in network.flows]
    link_ids = [link.Lid for link in network.links]

    average_utilities = []
    channel_data = {}  # Dict to hold data per channel per link

    # Collecting data
    for link in network.links:
        channel_data[link] = [0] * K  # Initialize with zero flow rate for each channel
        for user in network.users:
            if link in user.links:
                for flow in user.flows:
                    channel_data[link][flow.channel] += flow.rate

    for i, link in enumerate(network.links):
        rates = channel_data[link]
        utilities = [(rate / link.total_capacity) * 100 for rate in rates]  # Calculate utility as a percentage
        average_utility = sum(utilities)
        rounded_average_utility = int(round(average_utility))
        average_utilities.append(rounded_average_utility)

    # Calculate the overall average utility
    overall_average_utility = sum(average_utilities) / len(average_utilities)

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 7))  # 2 rows, 1 column

    # Subplot 1: Flow Rates vs. Data Amounts
    axs[0].scatter(data_amounts, flow_rates, color='red')
    axs[0].set_title('Flow Rates vs. Data Amounts')
    axs[0].set_xlabel('Data Amount')
    axs[0].set_ylabel('Flow Rate (bps)')
    axs[0].grid(True)

    # Subplot 2: Link Capacity Utilization
    colors = ['green' if percent == 100 else 'red' for percent in average_utilities]
    axs[1].bar(link_ids, average_utilities, color=colors, width=0.4)  # Adjust bar width to 0.4 for thinner bars
    axs[1].set_title('Link Capacity Utilization (%)')
    axs[1].set_xlabel('Link ID')
    axs[1].set_ylabel('Utilization (%)')
    axs[1].set_ylim(0, 100)
    axs[1].set_xticks(link_ids)
    axs[1].tick_params(axis='x', rotation=45)  # Rotate x-tick labels to prevent overlap
    axs[1].grid(True)

    # Adding a horizontal line for the overall average utility
    axs[1].axhline(y=overall_average_utility, color='blue', linestyle='--', label=f'Average Utility: {overall_average_utility:.2f}%')
    axs[1].legend(loc='upper left')

    # Display the combined plot
    plt.tight_layout()
    plt.show(block=False)
# Orya: change above

def divideToCluster(numOfUsersinCluster, linkStatus, links, BW_cluser, Ptol, omega):
    arr = []
    if numOfUsersinCluster == 2:
        if linkStatus == "Up":
            cluster1 = Cluster(1, [links[0], links[6]], BW_cluser, Ptol, omega)
            cluster2 = Cluster(2, [links[1], links[7]], BW_cluser, Ptol, omega)
            cluster3 = Cluster(3, [links[2], links[8]], BW_cluser, Ptol, omega)
            cluster4 = Cluster(4, [links[3], links[9]], BW_cluser, Ptol, omega)
            cluster5 = Cluster(5, [links[4], links[10]], BW_cluser, Ptol, omega)
            cluster6 = Cluster(6, [links[5], links[11]], BW_cluser, Ptol, omega)
            arr.append(cluster1)
            arr.append(cluster2)
            arr.append(cluster3)
            arr.append(cluster4)
            arr.append(cluster5)
            arr.append(cluster6)
            return arr
        elif linkStatus == "Down":
            cluster1 = Cluster(1, [links[0], links[11]], BW_cluser, Ptol, omega)
            cluster2 = Cluster(2, [links[1], links[10]], BW_cluser, Ptol, omega)
            cluster3 = Cluster(3, [links[2], links[9]], BW_cluser, Ptol, omega)
            cluster4 = Cluster(4, [links[3], links[8]], BW_cluser, Ptol, omega)
            cluster5 = Cluster(5, [links[4], links[7]], BW_cluser, Ptol, omega)
            cluster6 = Cluster(6, [links[5], links[6]], BW_cluser, Ptol, omega)
            arr.append(cluster1)
            arr.append(cluster2)
            arr.append(cluster3)
            arr.append(cluster4)
            arr.append(cluster5)
            arr.append(cluster6)
            return arr  

    elif numOfUsersinCluster == 3:
        if linkStatus == "Up":
            cluster1 = Cluster(1, [links[0], links[4], links[8]], BW_cluser, Ptol, omega)
            cluster2 = Cluster(2, [links[1], links[5], links[9]], BW_cluser, Ptol, omega)
            cluster3 = Cluster(3, [links[2], links[6], links[10]], BW_cluser, Ptol, omega)
            cluster4 = Cluster(4, [links[3], links[7], links[11]], BW_cluser, Ptol, omega)
            arr.append(cluster1)
            arr.append(cluster2)
            arr.append(cluster3)
            arr.append(cluster4)
            return arr
        elif linkStatus == "Down":
            cluster1 = Cluster(1, [links[0], links[4], links[11]], BW_cluser, Ptol, omega)
            cluster2 = Cluster(2, [links[1], links[5], links[10]], BW_cluser, Ptol, omega)
            cluster3 = Cluster(3, [links[2], links[6], links[9]], BW_cluser, Ptol, omega)
            cluster4 = Cluster(4, [links[2], links[7], links[8]], BW_cluser, Ptol, omega)
            arr.append(cluster1)
            arr.append(cluster2)
            arr.append(cluster3)
            arr.append(cluster4)
            return arr

    elif numOfUsersinCluster == 4:
        if linkStatus == "Up":
            cluster1 = Cluster(1, [links[0], links[3], links[6], links[9]], BW_cluser, Ptol, omega)
            cluster2 = Cluster(2, [links[1], links[4], links[7], links[10]], BW_cluser, Ptol, omega)
            cluster3 = Cluster(3, [links[2], links[5], links[8], links[11]], BW_cluser, Ptol, omega)
            arr.append(cluster1)
            arr.append(cluster2)
            arr.append(cluster3)
            return arr
        elif linkStatus == "Down":
            cluster1 = Cluster(1, [links[0], links[3], links[8], links[11]], BW_cluser, Ptol, omega)
            cluster2 = Cluster(2, [links[1], links[4], links[7], links[10]], BW_cluser, Ptol, omega)
            cluster3 = Cluster(3, [links[2], links[5], links[6], links[9]], BW_cluser, Ptol, omega)
            arr.append(cluster1)
            arr.append(cluster2)
            arr.append(cluster3)
            return arr

    raise ValueError(f"undefined_scenario:\n numOfUsersinCluster:{numOfUsersinCluster}, linkStatus:{linkStatus}")

def compare_OMA_NOMA_rates(network):
    # Parameters
    cluster_sizes = [2, 3, 4]  # User cluster sizes
    link_statuses = ['Up', 'Down']

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), sharey=True)

    for index, link_status in enumerate(link_statuses):
        OmaRate = network.Calculate_OMA_rates(link_status)
        Oma_rates = [OmaRate] * len(cluster_sizes)
        NomaRates = []
        for num_of_users in cluster_sizes:
            NomaRates.append(network.Calculate_NOMA_rates(num_of_users, link_status))

        # Plotting
        axes[index].plot(cluster_sizes, Oma_rates, 'r--', label='OMA Rate',
                         linewidth=2)  # OMA rates as a red dashed line
        axes[index].scatter(cluster_sizes, NomaRates, color='blue', label='NOMA Rates', s=100,
                            zorder=5)  # NOMA rates as points
        axes[index].set_title(f'{link_status} Link Rates')
        axes[index].set_xlabel('Number of Users in Cluster')
        axes[index].set_ylabel('Rate (Mbps)')
        axes[index].set_xticks(cluster_sizes)  # Set X-ticks to only include the cluster sizes
        axes[index].legend()
        axes[index].grid(True)

    plt.tight_layout()
    plt.show(block=False)

def calculate_path_loss(distance, path_loss_exponent=3.5, PL0=-30):
    """
    Calculate the path loss using the log-distance path loss model.
    :param distance: The distance between the transmitter and receiver in meters.
    :param path_loss_exponent: The path loss exponent, depending on the environment.
    :param PL0: The path loss at a reference distance (typically 1 meter) in dB.
    :return: The path loss in dB.
    """
    if distance <= 0:
        raise ValueError("Distance must be greater than 0")
    path_loss = PL0 + 10 * path_loss_exponent * np.log10(distance)
    return path_loss

def rayleigh_fading():
    """
    Simulate small scale fading using Rayleigh distribution.
    :return: A random fading coefficient sampled from a Rayleigh distribution.
    """
    return np.random.rayleigh()