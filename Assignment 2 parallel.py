# Now we parallelize the simulation to run the simulation quicker.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import simpy
import itertools
import scipy.stats as stats
import concurrent.futures
from itertools import repeat
import time
import tqdm
import os

class Bank:
    """
    A bank has a limited number of counters (``NUM_COUNTERS``) to serve customers in parallel.
    """

    def __init__(self, env, num_counters, service_rate, priority="FIFO", service_type = "M"):
        """
        Initialize the bank with a number of counters (``NUM_COUNTERS``) and a service rate (``MU``).
        priority: FIFO, SJF (shortest job first).

        If shortest job first is used, we require a priorityResource
        """
        self.env = env

        if priority == "FIFO":
            self.counter = simpy.Resource(env, num_counters)
        elif priority == "SJF":
            self.counter = simpy.PriorityResource(env, num_counters)
        else:
            raise ValueError(f"Priority {priority} not supported")

        self.priority = priority
        self.service_rate = service_rate
        self.customers_served = 0
        self.waiting_times = []
        self.service_times = []
        self.service_type = service_type
    
    def serve(self, customer, waiting_time, service_time):
        """
        The customer is served for a certain amount of time.
        """
        self.customers_served += 1
        self.waiting_times.append(waiting_time)
        self.service_times.append(service_time)
        yield self.env.timeout(service_time)

def setup(env, bank, SIMULATION_TIME, arrival_rate, print_all=False):
    """
    Create new customers until the SIMULATION_TIME reaches a certain value.
    Specify the customer function to use so that different queueing disciplines can be simulated.
    """
    customer_count = itertools.count()

    # Create more cars while the simulation is running
    while env.now < SIMULATION_TIME:
        yield env.timeout(random.expovariate(arrival_rate))
        env.process(customer(env, f'Customer {next(customer_count)}', bank, print_all=print_all))
    

def customer(env, name, bank, print_all=False):
    """
    Customer arrives, is served and leaves.
    priority: FIFO, SJF (shortest job first). 
    """
    arrive = env.now

    # Computer service time here, so it can later be used as priority
    if bank.service_type == "M":
        service_time = random.expovariate(bank.service_rate)
    elif bank.service_type == "D":
        service_time = 1.0 / bank.service_rate
    elif bank.service_type == "H":
        true_rate = bank.service_rate
        rate75 = true_rate * 2
        rate25 = true_rate / 2.5
        if random.random() < 0.75:
            service_time = random.expovariate(rate75)
        else:
            service_time = random.expovariate(rate25)
            
    if print_all:
        print(f"{env.now:.4f} {name}: Arrived (my service time is {service_time:.4f})")
        # print('%7.4f %s: Here I am' % (arrive, name))

    # Check what type of  request to make
    if bank.priority == "FIFO":
        request = bank.counter.request()
    elif bank.priority == "SJF":
        request = bank.counter.request(priority=service_time)

    with request as req:
        yield req
        wait = env.now - arrive

        if print_all:
            print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))

        yield env.process(bank.serve(name, wait, service_time))

        if print_all:
            print('%7.4f %s: Finished' % (env.now, name))

def run_simulation(num_servers, mu, load, alpha, CI_radius, service_type, priority, sim_time=500, initial_data_size=50, seed=42, print_results=False, decimals=4):
    """
    Function to run the simulation until the confidence interval is within the CI_radius.
    Returns a dictionary with the average waiting times, average number of customers served and average service times.	
    
    Notable parameter:
    - initial_data_size: number of repetitions to run before starting to check for the confidence interval
    """

    z = stats.norm.ppf(1-alpha/2) # 1.96 for 95% confidence

    random.seed(seed)
    avg_waiting_times = []
    avg_customers_served = []
    avg_service_times = []
    while True:
        env = simpy.Environment()
        bank = Bank(env, num_servers, mu, priority=priority, service_type=service_type)
        env.process(setup(env, bank, sim_time, load * num_servers * mu))
        env.run()
        avg_waiting_times.append(np.mean(bank.waiting_times))
        avg_customers_served.append(bank.customers_served)
        avg_service_times.append(np.mean(bank.service_times))
        if len(avg_waiting_times) < initial_data_size:
            continue
        S = np.std(avg_waiting_times, ddof=1)
        if z * S/np.sqrt(len(avg_waiting_times)) < CI_radius:
            break
    
    if print_results:
        print(f"M/{service_type}/{num_servers}, {priority} queue simulation with λ = {round(load * num_servers * mu, decimals)}, μ = {round(mu, decimals)}, ρ = {round(load, decimals)} ({len(avg_waiting_times)} repetitions to reach 95% confidence)")
        print(f"Customers served: {round(np.mean(avg_customers_served), decimals)}")
        print(f"Average waiting time: {round(np.mean(avg_waiting_times), decimals)} s")

    return {
        'avg_waiting_times': avg_waiting_times,
        'avg_customers_served': avg_customers_served,
        'avg_service_times': avg_service_times,
        'num_repetitions': len(avg_waiting_times)
    }

def one_sim(num_servers, mu, load, sim_time, seed):
    """
    Function to run the simulation once.
    Returns a dictionary with the average waiting times, average number of customers served and average service times.	
    """
    try:
        # num_servers, mu, load, seed = args
        random.seed(seed)
        env = simpy.Environment()
        bank = Bank(env, num_servers, mu)
        env.process(setup(env, bank, sim_time, load * num_servers * mu))
        env.run()
        return [np.mean(bank.waiting_times), bank.customers_served, np.mean(bank.service_times)]
    
    except Exception as e:
        print(f"Exception: {e}")
        return [0, 0, 0]


def run_simulation_parallel(num_servers, mu, load, alpha, CI_radius, service_type, priority, sim_time=500, initial_data_size=50, seed=42, print_results=False, decimals=4):
    """
    Function to run the simulation until the confidence interval is within the CI_radius. Parallelized for the initial_data_size repetitions.
    Returns a dictionary with the average waiting times, average number of customers served and average service times.	
    
    Notable parameter:
    - initial_data_size: number of repetitions to run before starting to check for the confidence interval
    """
    z = stats.norm.ppf(1-alpha/2) # 1.96 for 95% confidence

    random.seed(seed)

    avg_waiting_times = []
    avg_customers_served = []
    avg_service_times = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use executor.map to run the simulation in parallel

        results = executor.map(one_sim, repeat(num_servers), repeat(mu), repeat(load), repeat(sim_time), [seed + i for i in range(initial_data_size)])

        for result in results:
            avg_waiting_times.append(result[0])
            avg_customers_served.append(result[1])
            avg_service_times.append(result[2])
        
    while True:
        env = simpy.Environment()
        bank = Bank(env, num_servers, mu, priority=priority, service_type=service_type)
        env.process(setup(env, bank, SIMULATION_TIME, load * num_servers * mu))
        env.run()
        avg_waiting_times.append(np.mean(bank.waiting_times))
        avg_customers_served.append(bank.customers_served)
        avg_service_times.append(np.mean(bank.service_times))

        S = np.std(avg_waiting_times, ddof=1)
        if z * S/np.sqrt(len(avg_waiting_times)) < CI_radius:
            break
    
    if print_results:
        print(f"M/{service_type}/{num_servers}, {priority} queue simulation with λ = {round(load * num_servers * mu, decimals)}, μ = {round(mu, decimals)}, ρ = {round(load, decimals)} ({len(avg_waiting_times)} repetitions to reach 95% confidence)")
        print(f"Customers served: {round(np.mean(avg_customers_served), decimals)}")
        print(f"Average waiting time: {round(np.mean(avg_waiting_times), decimals)} s")

    return {
        'avg_waiting_times': avg_waiting_times,
        'avg_customers_served': avg_customers_served,
        'avg_service_times': avg_service_times,
        'num_repetitions': len(avg_waiting_times)
    }
    

# run_simulation_parallel(1, MU, LOAD, alpha, CI_radius, "M", "FIFO", print_results=True)
if __name__ == '__main__':  
    RANDOM_SEED = 42
    SIMULATION_TIME = 500
    initial_data_size = 50

    MU = 1/10 # Service rate
    LOAD = 0.99 # Load factor

    # We repeat until the average waiting time estimate is within 3 second of the true value (95% confidence)
    alpha = 0.05
    CI_radius = 3

    n_data_points = 20

    # Create data and figures folders if they don't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("figures"):
        os.makedirs("figures")

    for num_servers in [1, 2, 4]:

        normal_times = []
        parallel_times = []
        initial_repetitions = []
        
        for CI_radius in tqdm.tqdm(np.linspace(3, 0.25, n_data_points), desc=f"Running simulation for {num_servers} servers"):
            # Find how many repetitions are needed to reach the confidence interval
            results = run_simulation(num_servers, MU, LOAD, alpha, CI_radius, "M", "FIFO", initial_data_size=initial_data_size, print_results=False)
            repetitions = results['num_repetitions']
            
            # Use 90% of the repetitions as initial data size
            start = time.time()
            res_norm = run_simulation(num_servers, MU, LOAD, alpha, CI_radius, "M", "FIFO", initial_data_size=int(0.9*repetitions), print_results=False)
            end = time.time()
            res_par = run_simulation_parallel(num_servers, MU, LOAD, alpha, CI_radius, "M", "FIFO", initial_data_size=int(0.9*repetitions), print_results=False)
            parallel_time = time.time() - end
            normal_time = end - start

            normal_times.append(normal_time)
            parallel_times.append(parallel_time)
            initial_repetitions.append(int(0.9*repetitions))
        
        # Reset plot and Plot the results
        plt.clf()
        plt.plot(np.linspace(3, 0.1, n_data_points), normal_times, label="Normal")
        plt.plot(np.linspace(3, 0.1, n_data_points), parallel_times, label="Parallel")
        plt.xlabel("Confidence interval radius")
        plt.ylabel("Time (s)")
        plt.title(f"Time to reach 95% confidence for {num_servers} servers. \n(μ = {MU:.2f}, λ = {LOAD * num_servers * MU:.2f}, ρ = {LOAD:.2f}, single sim time = {SIMULATION_TIME})")
        plt.legend()

        plt.savefig(f"figures/parallel_CI_vs_time_{num_servers}_servers.png")

        # Plot the time against the initial data size
        plt.clf()
        plt.plot(initial_repetitions, normal_times, label="Normal")
        plt.plot(initial_repetitions, parallel_times, label="Parallel")
        plt.xlabel("Initial data size")
        plt.ylabel("Time (s)")
        plt.title(f"Time to reach 95% confidence for {num_servers} servers. \n(μ = {MU:.2f}, λ = {LOAD * num_servers * MU:.2f}, ρ = {LOAD:.2f}, single sim time = {SIMULATION_TIME})")
        plt.legend()

        plt.savefig(f"figures/parallel_initial_data_vs_time_{num_servers}_servers.png")

        # Plot the speedup against the initial data size
        plt.clf()
        plt.plot(initial_repetitions, [normal_times[i]/parallel_times[i] for i in range(len(normal_times))])
        plt.xlabel("Initial data size")
        plt.ylabel("Speedup")
        plt.title(f"Speedup for {num_servers} servers. \n(μ = {MU:.2f}, λ = {LOAD * num_servers * MU:.2f}, ρ = {LOAD:.2f}, single sim time = {SIMULATION_TIME})")

        plt.savefig(f"figures/parallel_initial_data_vs_speedup_{num_servers}_servers.png")

        # Save the data to a csv file
        df = pd.DataFrame({
            "CI_radius": np.linspace(3, 0.1, n_data_points),
            "normal_times": normal_times,
            "parallel_times": parallel_times,
            "initial_repetitions": initial_repetitions
        })

        df.to_csv(f"data/parallel_CI_vs_time_{num_servers}_servers.csv", index=False)

