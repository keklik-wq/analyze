import numpy as np
import matplotlib.pyplot as plt
import time

from goa import grasshopper_optimization
from kha import krill_herd_algorithm
from ssa import salp_swarm_algorithm


def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def run_experiment():
    # Параметры экспериментов
    test_functions = [
        ("Sphere", sphere),
        ("Rastrigin", rastrigin),
        ("Rosenbrock", rosenbrock)
    ]
    
    algorithms = [
        ("GOA", grasshopper_optimization),
        ("KHA", krill_herd_algorithm),
        ("SSA", salp_swarm_algorithm)
    ]
    
    params = {
        "dim": 10,
        "n_agents": [10, 15, 25],
        "max_iter": 300,
        "lb": -10,
        "ub": 10,
        "runs": 5  
    }
    
    results = {fn_name: {alg_name: {"fitness": [], "time": []} 
              for alg_name, _ in algorithms} for fn_name, _ in test_functions}
    
    for fn_name, fn in test_functions:
        for n in params["n_agents"]:
            print(f"\n=== Testing {fn_name} with {n} agents ===")
            
            for alg_name, alg in algorithms:
                run_fitness = []
                run_times = []
                
                for _ in range(params["runs"]):
                    start_time = time.time()
                    _, best_fitness, convergence = alg(
                        fn, params["dim"], n, params["max_iter"], params["lb"], params["ub"]
                    )
                    elapsed = time.time() - start_time
                    
                    run_fitness.append(best_fitness)
                    run_times.append(elapsed)
                    results[fn_name][alg_name]["fitness"].append(convergence)
                
                avg_time = np.mean(run_times)
                print(f"{alg_name}: Best fitness {np.min(run_fitness):.2e}, Time {avg_time:.2f}s")
                results[fn_name][alg_name]["time"].append(avg_time)
    
    
    test_functions_names = [el[0] for el in test_functions]
    test_algorithms_names = [el[0] for el in algorithms]
    plot_results(results, params, test_functions_names, test_algorithms_names)

def plot_results(results, params, test_functions, algorithms):
    plt.figure(figsize=(15, 10))
    
    for i, fn_name in enumerate(test_functions):
        plt.subplot(2, 2, i+1)
        
        for alg_name in algorithms:
            avg_convergence = np.mean(results[fn_name][alg_name]["fitness"], axis=0)
            plt.plot(avg_convergence, label=alg_name)
        
        plt.title(fn_name)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.yscale('log')
        plt.legend()
        plt.grid()
    
    plt.subplot(2, 2, 4)
    x = np.arange(len(params["n_agents"]))
    width = 0.25
    
    for i, alg_name in enumerate(algorithms):
        times = []
        for fn_name in test_functions:
            times.append(np.mean(results[fn_name][alg_name]["time"]))
        
        plt.bar(x + i*width, times, width, label=alg_name)
    
    plt.title("Average Computation Time")
    plt.xlabel("Number of agents")
    plt.ylabel("Time (s)")
    plt.xticks(x + width, params["n_agents"])
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("optimization_lecture/optimization_comparison.png")
    plt.show()


if __name__ == "__main__":
    
    # dim = 10      
    # n_grasshoppers = 10  
    # max_iter = 500
    # lb = -10      
    # ub = 10  

    # best_solution, best_fitness, curve = grasshopper_optimization(
    #     sphere, dim, n_grasshoppers, max_iter, lb, ub
    # )
    # print("\n--- Results ---")
    # print(f"Best solution: {best_solution}")
    # print(f"Best function value: {best_fitness}")
    # best_solution, best_fitness, curve = krill_herd_algorithm(
    #     sphere, dim, n_grasshoppers, max_iter, lb, ub
    # )    
    # print("\n--- Results ---")
    # print(f"Best solution: {best_solution}")
    # print(f"Best function value: {best_fitness}")
    # best_solution, best_fitness, curve = salp_swarm_algorithm(
    #     sphere, dim, n_grasshoppers, max_iter, lb, ub
    # )

    # print("\n--- Results ---")
    # print(f"Best solution: {best_solution}")
    # print(f"Best function value: {best_fitness}")
    run_experiment()
