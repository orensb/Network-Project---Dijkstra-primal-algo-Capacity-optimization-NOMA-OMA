# --------------------------------------Import--------------------------------------
import copy
import sys
import matplotlib.pyplot as plt
import random


# --------------------------------------Classes--------------------------------------
class User:
    """ this class represent a user in the network """
    def __init__(self, Uid, links=[], startVertex=None, endVertices=[], rate=0.001):
        self.Uid = Uid
        self.links = links
        self.rate = rate
        self.startVertex = startVertex
        self.endVertices = endVertices

    def addLink(self, link):
        self.links.append(link)


class Vertex:
    """ this class represent a vertex in the network """
    def __init__(self, Vid):
        self.Vid = Vid
        self.ShortestPath = {}
        self.distance = sys.maxsize

    def __str__(self):
        return str(self.Vid)


class Link:
    """ this class represent a link in the network """
    def __init__(self, Lid, capacity, startVertex, endVertex, LagrangianMultiplier=0.5, weight=1):
        self.Lid = Lid
        self.capacity = capacity
        self.startVertex = startVertex
        self.endVertex = endVertex
        self.LagrangianMultiplier = LagrangianMultiplier
        self.weight = weight


class Network:
    """this class represent a network"""
    def __init__(self, users=[], links=[], vertices=[]):
        self.users = users
        self.links = links
        self.vertices = vertices

    def getNeighbours(self, vertex):
        neighbours = []
        for link in self.links:
            if link.startVertex.Vid == vertex.Vid:
                neighbours.append(link.endVertex)
            elif link.endVertex.Vid == vertex.Vid:
                neighbours.append(link.startVertex)
        return neighbours

    def getLinkWeight(self, startVertex, endVertex):
        for link in self.links:
            if link.startVertex.Vid == startVertex.Vid and link.endVertex.Vid == endVertex.Vid:
                return link.weight
            elif link.startVertex.Vid == endVertex.Vid and link.endVertex.Vid == startVertex.Vid:
                return link.weight
        return -1

    def addLink(self, link):
        self.links.append(link)

    def addVertex(self, vertex):
        self.vertices.append(vertex)

    def UpdateNetworkPaths(self, AlgorithmName):
        for vertex in self.vertices:
            if AlgorithmName == "Dijkstra":
                _, PN, _, NH = DijkstraAlgorithm(self, vertex)
            elif AlgorithmName == "BellmanFord":
                _, PN, _, NH = BellmanFordAlgorithm(self, vertex)
            PrintNextHop(NH, vertex)

            for key in PN:
                vertex.ShortestPath[key.Vid] = []
                previousKey = None
                currKey = None
                # print("now we test key: ", key)
                First = True
                for each in PN[key]:
                    if First:
                        currKey = each
                        First = False
                    else:
                        previousKey = currKey
                        currKey = each
                    # print("previousKey: ", previousKey, " currKey: ", currKey)
                    if previousKey is not None and currKey is not None:
                        for link in self.links:
                            # print("link.startVertex.Vid: ", link.startVertex.Vid, " link.endVertex.Vid: ", link.endVertex.Vid)
                            if link.startVertex.Vid == previousKey.Vid and link.endVertex.Vid == currKey.Vid:
                                if link not in vertex.ShortestPath[key.Vid]:
                                    vertex.ShortestPath[key.Vid].append(link)
                                    # print("we added link: ", link.Lid)
                                break
                            elif link.startVertex.Vid == currKey.Vid and link.endVertex.Vid == previousKey.Vid:
                                if link not in vertex.ShortestPath[key.Vid]:
                                    vertex.ShortestPath[key.Vid].append(link)
                                    # print("we added link: ", link.Lid)
                                break

        for user in self.users:
            user.links = []
            for v in user.endVertices:
                for link in user.startVertex.ShortestPath[v.Vid]:
                    if link not in user.links:
                        user.links.append(link)


# --------------------------------------Functions--------------------------------------
def createDefaultNetwork(L):
    """ this function creates a default network with L+1 users and L links
    every user i that is not user 0 uses link i, and user 0 uses all the links """
    network = Network()
    totalCapacity = 0
    totalUsersOnLinks = 0
    for i in range(L+1):
        network.vertices.append(Vertex(Vid=i))
        if i == 0:
            network.users.append(User(Uid=0, startVertex=network.vertices[i]))
        else:
            network.links.append(Link(Lid=i-1, capacity=1, startVertex=network.vertices[i-1], endVertex=network.vertices[i]))
            network.users.append(User(Uid=i, links=[network.links[i-1]], startVertex=network.vertices[i-1], endVertices=[network.vertices[i]]))
            network.users[0].addLink(network.links[i-1])
            totalCapacity += network.links[i-1].capacity
            totalUsersOnLinks += 2
        if i == L:
            network.users[0].endVertices.append(network.vertices[i])
    for user in network.users:  # set the initial rate equal to the total capacity divided by the number of users
        user.rate = totalCapacity / totalUsersOnLinks
    #network.addLink(Link(Lid=L, capacity=1, startVertex=network.vertices[4], endVertex=network.vertices[0]))
    return network


def UpdateNetworkRates(network, alpha, AlgorithmName):
    """ this function update the rates of the users on a given network by a given algorithm
    and returns the rate of each user with graph of convergence """
    users = network.users
    xAxis = []
    yAxis = []
    for each in users:  # initialize the graph
        xAxis.append([])
        yAxis.append([])

    numOfIterations = 100000  # number of iterations to run the algorithm
    for i in range(numOfIterations):
        chosenUser = random.choice(users)  # choose a random user
        if AlgorithmName == "Primal":  # update the rate of the chosen user by the primal algorithm
            x_r = users[chosenUser.Uid].rate
            users[chosenUser.Uid].rate = CalcNextRatePrimal(chosenUser, users, alpha, x_r)
        elif AlgorithmName == "Dual":  # update the rate of the chosen user by the dual algorithm
            x_r = users[chosenUser.Uid].rate
            users[chosenUser.Uid].rate = CalcNextRateDual(chosenUser, users, alpha, x_r)
        xAxis[chosenUser.Uid].append(i)  # add the iteration number to the graph
        yAxis[chosenUser.Uid].append(chosenUser.rate)  # add the rate of the chosen user to the graph

    PrintResults(xAxis, yAxis, users, AlgorithmName, alpha)  # print the results and show the graph


def CalcNextRatePrimal(user, users, alpha, x_r, stepSize=0.0001):
    """ this function calculates the next rate of a given user for the primal algorithm """
    if alpha == "inf":
        maxUserLinks = 0
        for each in users:
            if len(each.links) > maxUserLinks:
                maxUserLinks = len(each.links)
        if len(user.links) == maxUserLinks:
            return min(1, x_r**0.999 + stepSize/10)
        else:
            return max(0, x_r**1.001 - stepSize/10)
    payment = 0
    for link in user.links:  # calculate the payment of the user
        rateSum = 0
        for u in users:  # calculate the sum of the rates of all the users on the link
            if link in u.links:
                rateSum += u.rate
        payment += penaltyFunction(rateSum, link.capacity)
    return stepSize * (pow(user.rate, -1*alpha) - payment) + x_r # calculate the next rate of the user


def penaltyFunction(rateSum, capacity):
    """ this function calculates the penalty function """
    if rateSum < capacity:
        return rateSum * capacity
    else:
        try:
            return pow(rateSum, 3) * 2
        except OverflowError:
            return 0


def PrintResults(xAxis, yAxis, users, AlgorithmName, alpha):
    """ this function prints the results of the algorithm """
    print(f"|----{AlgorithmName} Algorithm Results:-----|")
    for user in users:
        print(f"|user {user.Uid} rate : {user.rate} |")
    print(f"|---------------------------------|\n")
    for i in range(len(xAxis)):
        plt.plot(xAxis[i], yAxis[i], label=f"user {i}")
    plt.title(f"{AlgorithmName} Algorithm - default network, alpha={str(alpha)}")
    plt.xlabel("Iteration Number")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid()
    plt.show()


def CalcNextRateDual(user, users, alpha, x_r, stepSize=0.0001):
    """ this function calculates the next rate of a given user for the dual algorithm """
    if alpha == "inf":
        maxUserLinks = 0
        for each in users:
            if len(each.links) > maxUserLinks:
                maxUserLinks = len(each.links)
        if len(user.links) == maxUserLinks:
            return min(1, x_r**0.9991 + stepSize/10)
        else:
            return max(0, x_r**1.00111 - stepSize/10)
    Q_l = 0
    for link in user.links:  # calculate the payment of the user
        rateSum = 0  # Y_l
        for u in users:  # calculate the sum of the rates of all the users on the link
            if link in u.links:
                rateSum += u.rate

        if link.LagrangianMultiplier == 0:  # if the Lagrangian multiplier is 0
            L_delta = max(0, rateSum - link.capacity) * stepSize  # calculate the Lagrangian multiplier
        elif link.LagrangianMultiplier > 0:
            L_delta = (rateSum - link.capacity) * stepSize
        else:
            print("Lagrangian multiplier is negative")

        link.LagrangianMultiplier += L_delta  # update the Lagrangian multiplier of the link
        Q_l += link.LagrangianMultiplier
    return pow(Q_l, -1/alpha)  # the inverse function of the utilization function


def DijkstraAlgorithm(network, start_node):
    """ calc the shortest path from start_node to all other nodes in the network """
    unvisited_nodes = copy.copy(network.vertices)
    shortest_path = {}
    previous_nodes = {}
    max_value = sys.maxsize
    for node in unvisited_nodes:  # initialize the shortest path of all the nodes to infinity except the start node
        shortest_path[node] = max_value
    shortest_path[start_node] = 0

    while unvisited_nodes:  # while there are unvisited nodes
        current_min_node = None
        for node in unvisited_nodes:  # Iterate over the nodes
            if current_min_node is None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        neighbors = network.getNeighbours(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + network.calculate_link_weight()
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                previous_nodes[neighbor] = current_min_node  # We also update the best path to the current node

        unvisited_nodes.remove(current_min_node)

    nextHop = {}
    PN = {}
    for vertex in network.vertices:
        PN[vertex] = []
        if vertex != start_node:
            nextHop[vertex], PN[vertex] = FindNextHop(network.getNeighbours(start_node), vertex, previous_nodes,
                                                      start_node, Path=PN[vertex])
    return previous_nodes, PN, shortest_path, nextHop


def printDijkstraResult(previousNodes, shortestPath, start_node, target_node):
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        node = previousNodes[node]

    # Add the start node manually
    path.append(start_node)

    print("We found the following best path with a value of {}.".format(shortestPath[target_node]))
    for i in range(len(path) - 1, -1, -1):
        print(path[i], end=" ")
        if i != 0:
            print("->", end=" ")


def BellmanFordAlgorithm(network, start_node):
    """ calc the shortest path from start_node to all other nodes in the network """
    previousNodes = {}
    shortestDistance = {}
    for vertex in network.vertices:
        vertex.distance = sys.maxsize
        previousNodes[vertex] = start_node
    start_node.distance = 0
    shortestDistance[start_node] = 0

    for u in network.vertices:
        for v in network.getNeighbours(u):
            if u.distance != sys.maxsize and u.distance + network.calculate_link_weight() < v.distance:
                v.distance = u.distance + network.calculate_link_weight()
                shortestDistance[v] = v.distance
                previousNodes[v] = u
    nextHop = {}
    PN = {}
    for vertex in network.vertices:
        PN[vertex] = []
        if vertex != start_node:
            nextHop[vertex], PN[vertex] = FindNextHop(network.getNeighbours(start_node), vertex, previousNodes, start_node, Path=PN[vertex])
    return previousNodes, PN, shortestDistance, nextHop


def FindNextHop(neighbors, vertex, previousNodes, startNode, Path):
    """ this function finds the next hop of a given vertex """
    Path.append(vertex)
    if previousNodes[vertex] in neighbors:
        Path.append(previousNodes[vertex])
        Path.append(startNode)
        return previousNodes[vertex], Path
    elif previousNodes[vertex] == startNode:
        Path.append(startNode)
        return vertex, Path
    else:
        return FindNextHop(neighbors, previousNodes[vertex], previousNodes, startNode, Path)


def printBellmanFordResult(NextHop, shortestDistance, start_node, target_node):
    print("the shortest path from {} to {} is with length: {}".format(start_node.Vid, target_node.Vid, shortestDistance[target_node]))
    print("the next hop is: {}".format(NextHop[target_node].Vid))


def PrintUserLinks(users):
    for user in network1.users:
        print(f"user {user.Uid} links: \n", end="")
        for link in user.links:
            print(f"{link.Lid}", end="")
        print("\n")


def PrintTable(PN, vertex):
    print("Vertex: ", vertex.Vid, " PN: ")  # print the PN table of each vertex
    for each in PN:
        print(each, end=": ")
        for each2 in PN[each]:
            print(each2, end=" ")
        print()


def PrintNextHop(nextHop, vertex):
    print("Vertex: ", vertex.Vid, " NextHop: ")  # print the next hop table of each vertex
    for each in nextHop:
        print(each, ": ", nextHop[each])


# --------------------------------------Main--------------------------------------
if __name__ == "__main__":
    Links = 5
    alpha1 = "inf"     # 1-proportional fairness, 2-minimum potential delay fairness, infinity-max min fairness
    network1 = createDefaultNetwork(Links)
    #PrintUserLinks(network1.users)
    network1.UpdateNetworkPaths("BellmanFord")
    #PrintUserLinks(network1.users)
    #UpdateNetworkRates(network1, alpha1, "Primal")
    UpdateNetworkRates(network1, alpha1, "Primal")
    #shortest_path, previous_nodes = DijkstraAlgorithm(network1, network1.vertices[1])
    #printDijkstraResult(previous_nodes, shortest_path, network1.vertices[1], network1.vertices[6])
    #previousNodes, PN, shortestDistance, NextHop = BellmanFordAlgorithm(network1, network1.vertices[0])
    #printBellmanFordResult(NextHop, shortestDistance, network1.vertices[0], network1.vertices[6])