import numpy as np

from .cleark_weight import generate_random_vrp_data


def clarke_wright(time_matrix: np.ndarray, demand: np.ndarray, max_vehicle_capacity: np.ndarray, max_vehicle_time: np.ndarray):
    """
    Implementa o algoritmo de Clarke e Wright Savings para VRP com restrições de capacidade e tempo máximo por veículo.

    Args:
        time_matrix (np.ndarray): Matriz de tempo entre os clientes e o depósito.
        demand (np.ndarray): Demanda de cada cliente.
        max_vehicle_capacity (np.ndarray): Capacidade máxima de cada veículo.
        max_vehicle_time (np.ndarray): Tempo máximo que cada veículo pode operar.

    Returns:
        List[List[int]]: Lista de rotas, onde cada rota é uma lista de índices de clientes.
    """
    
    num_customers = len(demand)
    savings = []
    
    # Inicializar rotas individuais (cada cliente tem sua própria rota)
    routes = [[0, i, 0] for i in range(1, num_customers)]
    current_capacities = [demand[i] for i in range(1, num_customers)]
    current_times = [time_matrix[0, i] + time_matrix[i, 0] for i in range(1, num_customers)]
    
    # Calcula os savings para cada par de clientes (i, j)
    for i in range(1, num_customers):
        for j in range(i + 1, num_customers):
            saving = time_matrix[0, i] + time_matrix[0, j] - time_matrix[i, j]
            savings.append((i, j, saving))
    
    # Ordenar os savings em ordem decrescente
    savings.sort(key=lambda x: x[2], reverse=True)

    used_vehicles = [False] * len(max_vehicle_capacity)  # Controla veículos usados
    vehicle_routes = [[] for _ in range(len(max_vehicle_capacity))]  # Guarda as rotas de cada veículo

    for i, j, _ in savings:
        route_i = next((route for route in routes if i in route), None)
        route_j = next((route for route in routes if j in route), None)
        
        if route_i != route_j:
            # Tentar encontrar um veículo que possa atender a combinação de rotas
            for vehicle_idx in range(len(max_vehicle_capacity)):
                if not used_vehicles[vehicle_idx] or (used_vehicles[vehicle_idx] and vehicle_idx < len(vehicle_routes)):
                    new_time = current_times[routes.index(route_i)] + current_times[routes.index(route_j)] - time_matrix[i, 0] - time_matrix[0, j] + time_matrix[i, j]
                    
                    # Verificar se as rotas podem ser combinadas respeitando a capacidade e o tempo máximo do veículo
                    if current_capacities[routes.index(route_i)] + current_capacities[routes.index(route_j)] <= max_vehicle_capacity[vehicle_idx] and new_time <= max_vehicle_time[vehicle_idx]:
                        if route_i[-2] == i and route_j[1] == j:
                            # Obter os índices antes de remover as rotas
                            idx_i = routes.index(route_i)
                            idx_j = routes.index(route_j)

                            # Atualizar a rota i com os clientes da rota j
                            route_i[-1:] = route_j[1:]  # Unir as rotas
                            # Atualizar capacidade e tempo
                            current_capacities[idx_i] += current_capacities[idx_j]
                            current_times[idx_i] = new_time

                            # Remover a rota j
                            routes.pop(idx_j)
                            current_capacities.pop(idx_j)
                            current_times.pop(idx_j)

                            # Marcar o veículo como usado
                            used_vehicles[vehicle_idx] = True
                            vehicle_routes[vehicle_idx].append(route_i)
                            break
    
    # Retornar as rotas otimizadas para cada veículo
    return [route for route in routes if route]