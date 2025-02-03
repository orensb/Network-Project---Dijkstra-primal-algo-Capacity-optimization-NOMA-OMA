import time
import matplotlib.pyplot as plt
import fun_function
import Fun


def print_simulation_options():
    print(f"4: solving NUM problem with primal or dual algorithm with different alpha-fairness values")
    print(f"5: setting the path with Dijkstra and then divide rates with primal- dual algorithm")
    print(f"6: divide rates to flow relatively to data amount with interference channels")
    print(f"7: optimize the results in 6 by allowing routing flows to different channels")
    print(f"8: Simulate paper")
    print(f"99: stop simulating")


def main_menu_ui():
    print(f"\nMain-Menu")
    print_simulation_options()
    selection = input(f"\nplease choose from above:")
    while selection not in ["4", "5", "6", "7", "8", "99"]:
        print(f"\n\nyour input can only be from below options, please select again:")
        print_simulation_options()
        selection = input(f"\nselection:")
    return int(selection)


def get_user_inputs(param_names):
    # Initialize the dictionary to store valid inputs
    params = {}

    while True:
        set_default = input("Do you want to set all parameters to default? (y/n): ")
        if set_default.lower() == "y":
            # Set all parameters to default values and break the loop
            return params
        elif set_default.lower() == "n":
            for name in param_names:
                while True:
                    try:
                        # Prompt user for input
                        value = int(input(f"Enter a positive integer for {name.upper()}: "))
                        # Check if the value is greater than 0
                        if name.upper() == 'N' and value <= 1:
                            print("N must be greater than 1. Please try again.")
                        else:
                            params[name] = value
                            break
                    except ValueError:
                        # Handle cases where the conversion to int fails
                        print(f"Invalid input. {name.upper()} must be a positive integer.")
            return params
        else:
            print("Invalid input. Please enter either 'y' or 'n'.")


def set_user_inputs(param_names):
    user_inputs = get_user_inputs((param_names))
    Fun.set_global_params(**user_inputs)


def get_algorithm_input():
    while True:
        try:
            algo_input = int(input("Enter 1 for Primal algorithm or 2 for Dual algorithm: "))
            if algo_input == 1:
                return "Primal"
            elif algo_input == 2:
                return "Dual"
            else:
                print("Invalid input. Please enter either 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_alpha_input():
    while True:
        try:
            alpha_input = int(input("Enter 1 for alpha = 1, 2 for alpha = 2, or 3 for alpha = infinity: "))
            if alpha_input == 1:
                return 1
            elif alpha_input == 2:
                return 2
            elif alpha_input == 3:
                return float('inf')
            else:
                print("Invalid input. Please enter either 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

6
def get_num_of_users_input():
    while True:
        try:
            num_users_input = int(input("Enter the number of users in the cluster (2, 3, or 4): "))
            if num_users_input in [2, 3, 4]:
                return num_users_input
            else:
                print("Invalid input. Please enter either 2, 3, or 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def active_Q4():
    N = 5
    Algo = get_algorithm_input()
    alpha = get_alpha_input()
    Fun.set_global_params(n=N)
    Fun.calc_inter_face = False
    network = Fun.Network(create_network_type="NUM")
    network.draw_network()
    print(network)
    fun_function.CalcNetworkRate(network, alpha, Algo)
    plt.show()  # to stop the code so we can analyze the graph showing the rates


def active_Q5():
    N = 5  # num of users
    M = 10 # network radius
    R = 2  # neighbors radius
    set_user_inputs(["n", "m", "r"])
    Algo = get_algorithm_input()
    alpha = get_alpha_input()
    Fun.calc_inter_face = False
    network = Fun.Network()
    network.draw_network()

    print(f'\nResult before Dijkstra\n')
    fun_function.CalcNetworkRate(network, alpha, Algo)
    print(f'\nResult after Dijkstra\n')
    network.update_network_paths_using_Dijkstra()
    fun_function.CalcNetworkRate(network, alpha, Algo)
    plt.show()


def active_Q6():
    set_user_inputs(["n", "m", "r", "k"])
    Fun.calc_inter_face = True
    network = Fun.Network()
    network.update_network_paths_using_Dijkstra()
    fun_function.set_flows_rate_based_on_tdma(network, Fun.K)
    plt.show()


def active_Q7():
    set_user_inputs(["n", "m", "r", "k", "f"])
    Fun.calc_inter_face = True
    network = Fun.Network()
    network.update_network_paths_using_Dijkstra()
    fun_function.set_flows_rate_based_on_tdma(network, Fun.K)
    fun_function.optimize_flows_rate_based_on_tdma(network, Fun.K)
    plt.show()


def active_Q8():
    N = 12
    Fun.numOfUsersinCluster = get_num_of_users_input()
    assert Fun.numOfUsersinCluster in [2, 3, 4]
    Fun.set_global_params(n=N, k=N)
    network = Fun.Network(num_of_users=N, create_network_type="Random")
    network.draw_network()
    fun_function.compare_OMA_NOMA_rates(network)
    plt.show()

Q_active_functions_dict = {4: active_Q4, 5: active_Q5, 6: active_Q6, 7: active_Q7, 8: active_Q8}
title = f"Welcome to Network-Simulation x3000 by Oryan, Oren, Roni, Talya & Bar"

if __name__ == "__main__":
    print("-"*len(title))
    print(title)
    print("-" * len(title))
    main_menu_selection = main_menu_ui()
    while main_menu_selection != 99:
        Q_active_function = Q_active_functions_dict.get(main_menu_selection)
        Q_active_function()
        main_menu_selection = main_menu_ui()
    print(f"\n\nthank you for playing, bye-bye!")
    time.sleep(1.5)