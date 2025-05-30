import numpy as np

def krill_herd_algorithm(obj_func, dim, n_krill, max_iter, lb, ub):
    krill = np.random.uniform(lb, ub, (n_krill, dim))
    fitness = np.array([obj_func(k) for k in krill])
    best_idx = np.argmin(fitness)
    X_food = krill[best_idx].copy()
    leader_fitness = fitness[best_idx]
    
    N_max = 0.01  
    D_max = 0.002 
    w_n = 0.4     
    w_f = 0.9     
    
    N = np.zeros((n_krill, dim))  
    F = np.zeros((n_krill, dim))  
    
    convergence_curve = np.zeros(max_iter)
    for t in range(max_iter):
        for i in range(n_krill):
            neighbors = krill[np.random.choice(n_krill, size=5, replace=False)]
            N[i] = N_max * np.sum(neighbors - krill[i], axis=0) + w_n * N[i]
            
            F[i] = 0.8 * (X_food - krill[i]) + w_f * F[i]
            
            D_i = D_max * (1 - t/max_iter) * np.random.uniform(-1, 1, dim)
            
            krill[i] += (N[i] + F[i] + D_i)
            krill[i] = np.clip(krill[i], lb, ub)
        
        fitness = np.array([obj_func(k) for k in krill])
        new_best_idx = np.argmin(fitness)
        if fitness[new_best_idx] < leader_fitness:
            X_food = krill[new_best_idx].copy()
            leader_fitness = fitness[new_best_idx]
        
        convergence_curve[t] = leader_fitness
        if t % 50 == 0:
            print(f"Iteration {t}: Best Fitness = {leader_fitness}")
            #print(f"Iteration {t}: Curve = {convergence_curve}")
    
    return X_food, leader_fitness, convergence_curve

def sphere(x):
    return np.sum(x ** 2)


if __name__ == "__main__":
    
    dim = 10      
    n_grasshoppers = 10  
    max_iter = 500
    lb = -10      
    ub = 10       
    best_solution, best_value = krill_herd_algorithm(
        sphere, dim=10, n_krill=50, max_iter=500, lb=-10, ub=10
    )
    print("\n--- Results ---")
    print(f"Best solution: {best_solution}")
    print(f"Best function value: {best_value}")