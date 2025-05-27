import numpy as np

def salp_swarm_algorithm(obj_func, dim, n_salps, max_iter, lb, ub):
    """
    Параметры:
    - obj_func: целевая функция
    - dim: размерность задачи
    - n_salps: количество сальп в популяции
    - max_iter: максимальное число итераций
    - lb, ub: нижние и верхние границы поиска
    """
    
    salps = np.random.uniform(lb, ub, (n_salps, dim))
    fitness = np.array([obj_func(salp) for salp in salps])
    
    leader_idx = np.argmin(fitness)
    leader = salps[leader_idx].copy()
    leader_fitness = fitness[leader_idx]
    
    convergence_curve = np.zeros(max_iter)
    
    for t in range(max_iter):
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)
        
        for i in range(n_salps):
            if i == 0:  # leader
                for j in range(dim):
                    c2 = np.random.rand()
                    c3 = np.random.rand()
                    
                    if c3 >= 0.5:
                        salps[i, j] = leader[j] + c1 * ((ub - lb) * c2 + lb)
                    else:
                        salps[i, j] = leader[j] - c1 * ((ub - lb) * c2 + lb)
                    
                    
                    salps[i, j] = np.clip(salps[i, j], lb, ub)
            else:  # Ведомые сальпы
                salps[i] = 0.5 * (salps[i] + salps[i - 1])
        
        fitness = np.array([obj_func(salp) for salp in salps])
        
        new_leader_idx = np.argmin(fitness)
        if fitness[new_leader_idx] < leader_fitness:
            leader = salps[new_leader_idx].copy()
            leader_fitness = fitness[new_leader_idx]
        
        convergence_curve[t] = leader_fitness
        
        if t % 50 == 0:
            print(f"Iteration {t}: Best Fitness = {leader_fitness}")
            #print(f"Iteration {t}: Curve = {convergence_curve}")
    
    return leader, leader_fitness, convergence_curve
    #return leader, leader_fitness

def sphere(x):
    return np.sum(x ** 2)

if __name__ == "__main__":
    
    dim = 10      
    n_salps = 50  
    max_iter = 500
    lb = -10      
    ub = 10       

    best_solution, best_fitness, convergence = salp_swarm_algorithm(
        sphere, dim, n_salps, max_iter, lb, ub
    )

    print("\n--- Results ---")
    print(f"Best solution: {best_solution}")
    print(f"Best function value: {best_fitness}")