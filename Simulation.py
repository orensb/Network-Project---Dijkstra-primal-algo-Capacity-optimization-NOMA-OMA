# --------------------------------------Import--------------------------------------
import random
import math
import matplotlib.pyplot as plt
import copy
import fun_function
# --------------------------------------Defines--------------------------------------
# Orya: change below
N = 6  # num of vertex
M = 10  # network radius
R = 2  # neighbors radius
K = 1  # num of channels
F = 10  # number of flows
FLOW_MAX_DATA = 1000
calc_inter_face = False

CONSTANT_POWER = 1
CONSTANT_BW = 1
down_Ri = 41e5
K_B = 1.38e-23  # Boltzmann constant in Joules/Kelvin
T = 290       # Temperature in Kelvin


def set_global_params(k=None, n=None, m=None, r=None, f=None):
    global K, N, M, R, F
    K = k or 1
    N = n or 5
    M = m or 10
    R = r or 2
    F = f or 10
# Orya: change above
# --------------------------------------Classes--------------------------------------
class Vertex:
    """ this class represent a vertex in the network """

    def __init__(self, Vid):
        self.Vid = Vid
        self.location = self.generate_location()
        self.neighbors = {}
        self.ShortestPath = {}  # Dictionary to store shortest paths to other vertices
        self.power = 0
        self.bw = 0

    def __str__(self):
        return str(self.Vid)

    def __lt__(self, other):
        return self.Vid < other.Vid

    def __eq__(self, other):
        return self.Vid == other.Vid

    def __hash__(self):
        return hash(self.Vid)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except the dictionary to avoid deep copying issues
        result.Vid = self.Vid
        result.location = copy.deepcopy(self.location, memo)
        result.power = self.power
        result.bw = self.bw

        # Now handle the dictionary carefully
        result.neighbors = {copy.deepcopy(key, memo): copy.deepcopy(value, memo) for key, value in
                            self.neighbors.items()}
        result.ShortestPath = {copy.deepcopy(key, memo): copy.deepcopy(value, memo) for key, value in
                               self.ShortestPath.items()}
        return result

    def generate_location(self):
        """
        Generates a random location within a circle of radius M centered at the origin.
        """
        theta = random.uniform(0, 2 * math.pi)  # Angle for circular distribution
        r = M * math.sqrt(
            random.uniform(0, 1))  # Distance from the center, sqrt for uniform distribution within the circle
        x = round(r * math.cos(theta), 2)
        y = round(r * math.sin(theta), 2)
        return x, y

    def distance_to(self, other):
        """
        Calculates the Euclidean distance between two vertices.
        """
        x1, y1 = self.location
        x2, y2 = other.location
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return distance

    def calc_neighbors(self, others, r):
        '''
        loops over other vertices, calculates their distance to the current vertex,
        and adds them to the list if they are within the radius.
        '''
        neighbors = []
        for other in others:
            distance = self.distance_to(other)
            if 0 < distance <= r:
                neighbors.append((other, distance))

        return neighbors

    def add_neighbors(self, neighbor_vertex, connected_link):
        '''
        Adds a neighboring vertex to the current vertex's neighbor list
        and associates it with a connected_link.
        '''
        self.neighbors[neighbor_vertex] = connected_link


class User:
    def __init__(self, Uid, links=[], startVertex=None, endVertex=None, rate=0.001):
        self.Uid = Uid
        self.links = links
        self.defualtLinks = []
        self.rate = rate
        self.startVertex = startVertex
        self.endVertex = endVertex
        self.flows = []

    def __str__(self):
        string = f"\nUser({self.Uid}) connect {self.startVertex.Vid} to {self.endVertex.Vid} through: "
        for link in self.links:
            string += f"{link} "
        string += f", and sends {len(self.flows)} flow with {sum(flow.data_amount for flow in self.flows)} data"
        return string

    def add_flow(self,flow):
        self.flows.append(flow)


class Link:
    """ this class represent a link in the network """

    def __init__(self, Lid, vertex1: Vertex = None, vertex2: Vertex = None, LagrangianMultiplier=0.5, distance=None):
        self.Lid = Lid
        self.connected_vertices = (vertex1, vertex2)
        self.distance = distance
        self.LagrangianMultiplier = LagrangianMultiplier
        self.gain = self.calculate_gain()
        self.power = self.calculate_link_power()
        self.interference_gain = self.calculate_interference_gain()
        self.interference_power = 0
        self.total_capacity = self.calculate_capacity()
        self.channels_capacities = [self.total_capacity / K for _ in range(K)]
        self.weight = self.calculate_link_weight()


    def __str__(self):
        return str(self.Lid)

    def calculate_gain(self):
        '''
        Calculates the link's gain based on the distance between 2 vertices.
        '''
        distance = self.distance
        path_loss_db = fun_function.calculate_path_loss(distance)
        fading_coefficient = fun_function.rayleigh_fading()
        attenuation_factor = 10 ** (path_loss_db / 10)
        gain = attenuation_factor * fading_coefficient
        return gain

    def calculate_interference_gain(self):
        '''
        Calculates the interference gain for the link, which is similar to the
        gain but with a smaller constant to model interference.
        '''
        distance = self.distance
        path_loss_db = fun_function.calculate_path_loss(distance)
        fading_coefficient = fun_function.rayleigh_fading()
        attenuation_factor = 10 ** (path_loss_db / 10)
        gain = attenuation_factor * fading_coefficient*1e-23
        return gain

    def calculate_link_power(self):
        '''
        Calculates the power of the link. The power is the minimum power
        of the two connected vertices, scaled by the square of the gain.

        '''
        min_vertex_power = min(self.connected_vertices[0].power, self.connected_vertices[1].power)
        power = min_vertex_power * (self.gain ** 2)
        return power

    def calculate_capacity(self):
        '''
        Calculates the link's data capacity based on:
        BW, SINR
        '''
        bw = min(self.connected_vertices[0].bw, self.connected_vertices[1].bw)
        noise_power = K_B * T * bw
        SINR = self.power / (noise_power + self.interference_power)  # Simplified SINR calculation
        capacity = bw * math.log2(1 + SINR)
        if capacity <= 0:
            raise ValueError(f"Invalid capacity {capacity} for link {self.Lid}. Capacity must be greater than zero.")
        return capacity

    def calculate_link_weight(self):
        ''''
        link's weight based on the total capacity
        '''
        if self.total_capacity == 0:
            return float('inf')
        return 1 / self.total_capacity

    def update_total_capacity(self):
        self.total_capacity = sum(self.channels_capacities)


class Flow:
    def __init__(self, source=None, destination=None, data_amount=0, rate=0):
        self.source = source
        self.destination = destination
        self.data_amount = data_amount
        self.generate_flow()
        self.rate_by_links = {}
        self.rate = rate
        self.channel = random.randint(0, K-1)

    def __str__(self):
        string = f"{self.source} send to {self.destination} {self.data_amount}"
        return string

    def generate_flow(self):
        """Generates F random information flows."""
        source, destination = random.sample(range(1, N + 1), 2)
        data_amount = random.randint(1, FLOW_MAX_DATA)  # random data amount between 1 and 1000 units
        self.source = self.source or source
        self.destination = self.destination or destination
        self.data_amount = self.data_amount or data_amount

    def set_rate_2_min_of_rate_by_links(self):
        self.rate = min(self.rate_by_links.values())


class Cluster:
    def __init__(self, Cid, links=[], BW_cluser=None, Ptol=None, omega=None):
        self.Cid = Cid
        self.links = links
        self.vertices = [link.connected_vertices[0] for link in self.links]
        self.BW_cluser = BW_cluser
        self.Ptol = Ptol
        self.omega = omega

    def __str__(self):
        return str(self.links)

    def power_allocation(self, numOfUsersinCluster, linkStatus, Pt):
        '''
        allocates power to the vertices based on the number of users in the cluster,
        '''
        if numOfUsersinCluster == 2:
            if linkStatus == "Down":
                gains = []
                for link in self.links:
                    gains.append(link.gain)
                gamma1 = gains[:1][0]
                self.vertices[0].power = (Pt / 2) - (self.Ptol / (2 * gamma1))
                self.vertices[1].power = (Pt / 2) + (self.Ptol / (2 * gamma1))
                return

            if linkStatus == "Up":
                gains = []
                for link in self.links:
                    gains.append(link.gain)
                gamma1, gamma2 = gains
                self.vertices[0].power = Pt
                self.vertices[1].power = Pt
                return

        if numOfUsersinCluster == 3:
            if linkStatus == "Down":
                gains = []
                for link in self.links:
                    gains.append(link.gain)
                gamma1, gamma2 = gains[:2]
                self.vertices[0].power = (Pt / 4) - (self.Ptol / (2 * gamma1)) - (self.Ptol / (4 * gamma2))
                self.vertices[1].power = (Pt / 4) + (self.Ptol / (2 * gamma1)) - (self.Ptol / (4 * gamma2))
                self.vertices[2].power = (Pt / 2) + (self.Ptol / (2 * gamma2))
                return

            if linkStatus == "Up":
                gains = []
                for link in self.links:
                    gains.append(link.gain)
                gamma1, gamma2, gamma3 = gains
                self.vertices[0].power = Pt
                self.vertices[1].power = Pt
                self.vertices[2].power = Pt
                return

        if numOfUsersinCluster == 4:
            if linkStatus == "Down":
                gains = []
                for link in self.links:
                    gains.append(link.gain)
                gamma1, gamma2, gamma3 = gains[:3]
                self.vertices[0].power = (Pt / 8) - (self.Ptol / (2 * gamma1)) - (self.Ptol / (4 * gamma2)) - (self.Ptol / (8 * gamma3))
                self.vertices[1].power = (Pt / 8) + (self.Ptol / (2 * gamma1)) - (self.Ptol / (4 * gamma2)) - (self.Ptol / (8 * gamma3))
                self.vertices[2].power = (Pt / 4) + (self.Ptol / (2 * gamma2)) - (self.Ptol / (4 * gamma3))
                self.vertices[3].power = (Pt / 2) + (self.Ptol / (2 * gamma3))
                return

            if linkStatus == "Up":
                gains = []
                for link in self.links:
                    gains.append(link.gain)
                gamma1, gamma2, gamma3, gamma4 = gains
                self.vertices[0].power = Pt
                self.vertices[1].power = Pt
                self.vertices[2].power = Pt
                self.vertices[3].power = Pt
                return

        raise ValueError(f"undefined_scenario:\n numOfUsersinCluster:{numOfUsersinCluster}, linkStatus:{linkStatus}")


    def calculate_user_powers(self, Pt, LinkStatus):
        ''''
        selects the appropriate Ri calculation based on the link status 
        returns the calculated values.
        '''
        Link_functions = {"Up": self.calculate_user_powers_uplink, "Down": self.calculate_user_powers_downlink}
        CalcPower = Link_functions.get(LinkStatus)
        CalcPower(Pt)

    def calculate_down_link_Ri(self, numOfUsersinCluster):
        Ri_values = []
        for i in range(0, numOfUsersinCluster):
            denominator=0
            for j in range(1 , i-1):
                denominator = denominator + (self.vertices[j].power * self.links[i].gain)
            denominator +=self.omega
            Ri = down_Ri - self.omega * self.BW_cluser * math.log2(1 + ((self.vertices[i].power * self.links[i].gain) / denominator))
            Ri_values.append(Ri)
        return Ri_values

    def calculate_up_link_Ri(self, numOfUsersinCluster):
        Ri_values = []
        for i in range(0, numOfUsersinCluster):
            denominator = sum([self.vertices[j].power * self.links[j].gain for j in range(i + 1, numOfUsersinCluster)]) + self.omega
            Ri = (self.omega * self.BW_cluser * math.log2(1 + (self.vertices[i].power * self.links[i].gain / denominator)))/2
            Ri_values.append(Ri)
        return Ri_values

    def calculate_Ri(self, numOfUsersinCluster, LinkStatus):
        algorithm_functions = {"Up": self.calculate_up_link_Ri, "Down": self.calculate_down_link_Ri}
        CalcRi = algorithm_functions.get(LinkStatus)
        return CalcRi(numOfUsersinCluster)


class Network:
    """this class represent a network"""

    def __init__(self, num_of_users=N, radius=M, neighbors_radius=R, create_network_type="Random"):
        self.num_of_users = num_of_users
        self.radius = radius
        self.neighbors_radius = neighbors_radius
        self.users = []
        self.links = []
        self.vertices = []
        self.cluster = []
        self.flows = []
        self.PtUpLink = 24e-3
        self.PtDownLink = 46e-3
        self.systemBandwidth = 20e6
        self.network_type = create_network_type
        if create_network_type == "Random":
            self.create_network()
            self.generate_random_flows_and_users()
            self.calculate_interference_power()
        elif create_network_type == "NUM":
            self.create_NUM_network()
        elif create_network_type == "Paper":
            self.create_paper_network()
        else:
            raise ValueError(f"no network of type {create_network_type}, only: Random, NUM, Paper")

    def __str__(self):
        string_to_print = ""
        vertices_to_print = self.vertices
        string_to_print += f"\n{self.network_type} network general params:\n"
        string_to_print += f"num of users= {self.num_of_users}, network radius= {self.radius}, neighbors radius={self.neighbors_radius}\n"
        string_to_print += f"\nUSERS:"
        for user in self.users:
            string_to_print += f"{user}"
        string_to_print += f"\n\nFLOWS:\n"
        for flow in self.flows:
            string_to_print += f"{flow}\n"
        return string_to_print

    def create_NUM_network(self):
        previous_vertex = None
        for i in range(self.num_of_users):
            new_vertex = self.create_new_vertex(i)
            if previous_vertex is not None:  # If there's a previous vertex, connect it to the current one
                distance = previous_vertex.distance_to(new_vertex)
                new_link = self.create_new_link(previous_vertex, new_vertex, distance)
                self.create_neighbors(previous_vertex, new_vertex, new_link)
            previous_vertex = new_vertex
        self.generate_flows_and_users_NUM()
        for link in self.links :
            link.total_capacity = 1

    def create_paper_network(self):
        SmallGap = 0.0001
        for i in range(self.num_of_users):
            self.create_new_vertex(i)
        ConnectStation = self.create_new_vertex(N)
        BaseStation = self.create_new_vertex(N+1)
        BaseStation.location = (ConnectStation.location[0]+SmallGap,ConnectStation.location[1]+SmallGap)
        link = self.create_new_link(ConnectStation, BaseStation, ConnectStation.distance_to(BaseStation))
        self.links.remove(link)
        self.create_neighbors(ConnectStation, BaseStation, link)
        for i in range(self.num_of_users):
            user = self.vertices[i]
            link = self.create_new_link(user, ConnectStation, user.distance_to(BaseStation))
            self.create_neighbors(user, ConnectStation, link)

    def create_network(self):
        for i in range(self.num_of_users):
            self.create_new_vertex(i)
        for vertex in self.vertices:
            neighbors = vertex.calc_neighbors(self.vertices, self.neighbors_radius)
            for neighbor in neighbors:
                if neighbor[0] in vertex.neighbors:
                    continue
                link = self.create_new_link(vertex, neighbor[0], neighbor[1])
                self.create_neighbors(vertex, neighbor[0], link)

        if not self.check_network_connectivity():
            global R
            R += 1
            self.initial_network()


    def initial_network(self):
        self.neighbors_radius = R
        self.links = []
        self.vertices = []
        self.users = []
        self.create_network()

    def initial_users_rates(self):
        for user in self.users:
            user.rate = 0.001

    def initial_users(self):
        self.initial_users_rates()
        for user in self.users:
            user.links = user.defualtLinks

    def check_network_connectivity(self):
        remaining_vertices = self.vertices.copy()
        current_vertex = remaining_vertices[0]
        remaining_vertices = self.delete_neighbors(current_vertex, remaining_vertices)

        return remaining_vertices == []

    def delete_neighbors(self, vertex, remaining_vertices):
        remaining_vertices.remove(vertex)
        for neighbor in vertex.neighbors.keys():
            if neighbor not in remaining_vertices:
                continue
            self.delete_neighbors(neighbor, remaining_vertices)
        return remaining_vertices

    def calculateUserLinks(self, vertex, destVertex, visited=None):
        links = []
        if visited is None:
            visited = []
        visited.append(vertex)
        if destVertex in vertex.neighbors:
            links.append(vertex.neighbors[destVertex])
        else:
            for neighbor in vertex.neighbors:
                if neighbor in visited:
                    continue
                sub_links = self.calculateUserLinks(neighbor, destVertex, visited)
                if sub_links:
                    links.append(vertex.neighbors[neighbor])
                    links.extend(sub_links)
                    break
        return links

    def get_active_links(self):
        active_links = set()  # Use a set to avoid duplicates

        # Loop through all users to check their links
        for user in self.users:
            for link in user.links:
                active_links.add(link)  # Add each link to the set

        return list(active_links)

    def create_new_user(self, start_vertex, end_vertex):
        if start_vertex == end_vertex:
            raise ValueError(f"start vertex {start_vertex} cant be equal to end vertex {end_vertex}")
        userLinks = self.calculateUserLinks(start_vertex, end_vertex)
        Uid = len(self.users)+1
        user = User(Uid, links=userLinks, startVertex=start_vertex, endVertex=end_vertex)
        user.defualtLinks = userLinks
        self.users.append(user)
        return user

    def create_new_vertex(self, id):
        new_vertex = Vertex(Vid=id + 1)
        self.power_allocation(new_vertex)
        self.bw_aloocation(new_vertex)
        self.vertices.append(new_vertex)
        return new_vertex

    def create_new_link(self, vertex1, vertex2, distance):
        new_link = Link(Lid=f"{vertex1.Vid}-{vertex2.Vid}", vertex1=vertex1, vertex2=vertex2,
                        distance=distance)
        self.links.append(new_link)
        return new_link

    def create_neighbors(self, vertex1, vertex2, link):
        vertex1.add_neighbors(vertex2, link)
        vertex2.add_neighbors(vertex1, link)

    def sort_links_by_distance(self):
        self.links.sort(key=lambda x: x.distance)

    def power_allocation(self, vertex):  # TODO: maybe do a more sophisticated power allocation
        vertex.power = CONSTANT_POWER

    def bw_aloocation(self, vertex):  # TODO: maybe do a more sophisticated bw allocation
        vertex.bw = CONSTANT_BW

    def get_user(self, start_vertex, end_vertex):
        users = [user for user in self.users if user.startVertex == start_vertex and user.endVertex == end_vertex]
        if len(users) > 1:
            raise ValueError(f"there is unique user connect 2 vertex but got {len(users)} for vertx {start_vertex.Vid}")
        return users[0] if users else None

    def get_vertex(self, Vid):
        """Retrieve a vertex by its Vid."""
        for vertex in self.vertices:
            if vertex.Vid == Vid:
                return vertex
        raise ValueError(f"No vertex found with Vid {Vid}. vertices are from 1 to {N}")

    def create_Flow_and_User(self, idx1=None, idx2=None):
        flow = Flow(source=idx1, destination=idx2)
        source = self.get_vertex(flow.source)
        dest = self.get_vertex(flow.destination)
        self.flows.append(flow)
        user = self.get_user(source, dest) or self.create_new_user(source, dest)
        user.add_flow(flow)

    def generate_flows_and_users_NUM(self):
        self.flows, self.users = [], []

        # Connect first vertex to the last
        self.create_Flow_and_User(1, len(self.vertices)) # Assume min(Vid) == 1, there is no Vid == 0

        for i in range(1, len(self.vertices)):  # Connect each vertex to his follower expect for the first
            self.create_Flow_and_User(i, i+1)

    def generate_random_flows_and_users(self):
        self.flows, self.users = [], []
        for _ in range(F):
            self.create_Flow_and_User()

    def generate_flows_and_users_PrimalDual(self):  # TODO: currently not in use
        self.flows, self.users = [], []
        used_pairs = set()
        for _ in range(self.num_of_users):
            flow = Flow()
            source = self.get_vertex(flow.source)
            dest = self.get_vertex(flow.destination)
            pair = (source, dest)
            while pair in used_pairs:
                flow = Flow()
                source = self.get_vertex(flow.source)
                dest = self.get_vertex(flow.destination)
                pair = (source, dest)
            used_pairs.add(pair)
            self.create_Flow_and_User(source=flow.source, dest=flow.destination)

    def draw_network(self):
        fig, ax = plt.subplots()

        for vertex in self.vertices:
            x, y = vertex.location
            ax.plot(x, y, 'o', label=f'Vertex {vertex.Vid}')
            ax.text(x, y, f' {vertex.Vid}', verticalalignment='bottom', horizontalalignment='right')

        # Draw links
        for link in self.links:
            v1, v2 = link.connected_vertices
            x_values = [v1.location[0], v2.location[0]]
            y_values = [v1.location[1], v2.location[1]]
            ax.plot(x_values, y_values, 'k-', linewidth=0.5)  # 'k-' for black line

        ax.set_aspect('equal', 'box')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title(f'{self.network_type} Network Visualization')
        plt.show(block=False)

    def update_network_paths_using_Dijkstra(self, debug_prints=False):
        """Updates shortest paths for all vertices using Dijkstra's algorithm."""
        for v in self.vertices:
            v.ShortestPath = {}
        for start_vertex in self.vertices:
            distances, previous_nodes = fun_function.dijkstra_algorithm(self, start_vertex)
            for vertex in self.vertices:
                user = self.get_user(start_vertex, vertex)
                path, link_path = [], []
                current = vertex
                while current != start_vertex:
                    if current is None:
                        path, link_path = [], []
                        break
                    next_vertex = previous_nodes[current]
                    connected_link = current.neighbors[next_vertex]
                    link_path.append(connected_link)
                    path.append(current)
                    current = next_vertex
                path.reverse()
                link_path.reverse()
                start_vertex.ShortestPath[vertex] = path
                if user is not None:
                    if debug_prints and user.links != link_path:
                        print(f"\nupdate user({user.Uid}) links:")
                        print("from:", *[str(vertex) for vertex in user.links])
                        print("to  :", *[str(vertex) for vertex in link_path])
                    user.links = link_path

    def Calculate_NOMA_rates(self, numOfUsersinCluster, LinkStatus):
        self.sort_links_by_distance()
        gain = 40
        for link in self.links:
            if link.Lid == f"{self.num_of_users+1}-{self.num_of_users+2}":
                continue
            link.gain = gain
            gain = gain - 3

        BW_cluser = self.systemBandwidth/(N/numOfUsersinCluster)
        print(f"BW of cluster {BW_cluser}")
        Ptol = 10e-3
        omega = N/numOfUsersinCluster
        clusters_to_append = fun_function.divideToCluster(numOfUsersinCluster, LinkStatus, self.links, BW_cluser, Ptol, omega)
        self.cluster = []
        self.cluster += clusters_to_append
        clusters_Ri = []
        total_sum_Ri = 0
        "Go over each cluster, calculate the power and Ri"
        for cluster in self.cluster:
            print(f"Cluster id: {cluster.Cid}")
            Pt = self.PtUpLink if LinkStatus == "Up" else self.PtDownLink
            cluster.power_allocation(numOfUsersinCluster, LinkStatus, Pt)
            # for i in range(len(cluster.links)):
            #     print(f"gain {cluster.links[i].gain} ; power link: {cluster.vertices[i].power}")
            cluster_Ri = cluster.calculate_Ri(numOfUsersinCluster, LinkStatus)
            sum_Ri = sum(cluster_Ri)
            total_sum_Ri = total_sum_Ri + sum_Ri
            clusters_Ri.append(cluster_Ri)
            print(f"Ri of the cluster {[round(num,2) for num in cluster_Ri]}")
            print(f"Sum Ri of this cluster {round(sum_Ri,2)}")

        print(f"For {LinkStatus}Link and {numOfUsersinCluster} useer in cluster\nThe total Ri is:{round(total_sum_Ri,2)}")
        return  total_sum_Ri

    def Calculate_OMA_rates(self,LinkStatus):
        self.sort_links_by_distance()
        gain = 40
        for link in self.links:
            if link.Lid == f"{self.num_of_users+1}-{self.num_of_users+2}":
                continue
            link.gain = gain
            gain = gain - 3

        BW_oma = self.systemBandwidth/N
        if LinkStatus=="Down":
            power_user = self.PtDownLink/N
        elif LinkStatus=="Up":
            power_user = self.PtUpLink/N
        else:
            raise ValueError(f"undefined_scenario:\n linkStatus:{LinkStatus}")
        Ri_users=[]
        total_sum_Ri_oma=0
        N0=1
        for i in range(0,N):
            SNRi = power_user * self.links[i].gain
            SNRj=0
            for j in range(0,N):
                if (i !=j):
                    SNRj += power_user * self.links[j].gain
            Ri_oma = (BW_oma*math.log2(1+(SNRi/(N0+SNRj))))*10
            total_sum_Ri_oma +=Ri_oma
            Ri_users.append(Ri_oma)

        print(f"Ri of each user {Ri_users}")
        print(f"Total Ri for {total_sum_Ri_oma}")
        return total_sum_Ri_oma

    def calculate_interference_power(self):
        if not calc_inter_face:
            return
        active_links = set()
        for user in self.users:
            for link in user.links:
                active_links.add(link)
        for link in self.links:
            total_interference_power = 0
            for other_link in self.links:
                if other_link == link or other_link not in active_links:
                    continue

                interference_contribution = other_link.power * other_link.interference_gain
                total_interference_power += interference_contribution

            link.interference_power = total_interference_power
            link.total_capacity = link.calculate_capacity()







