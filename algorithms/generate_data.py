# Função para gerar dados aleatórios
import numpy as np


def generate_random_vrp_data(num_customers, num_vehicles, max_time_value=50, max_demand=10, max_capacity=30, max_time_limit=120):
    """
    Gera dados aleatórios para o problema de VRP.
    
    Args:
        num_customers (int): Número de clientes (excluindo o depósito).
        num_vehicles (int): Número de veículos.
        max_time_value (int): Tempo máximo entre dois pontos na matriz de tempo.
        max_demand (int): Demanda máxima que um cliente pode ter.
        max_capacity (int): Capacidade máxima de um veículo.
        max_time_limit (int): Tempo máximo que um veículo pode operar.

    Returns:
        time_matrix (np.ndarray): Matriz de tempo entre o depósito e os clientes.
        demand (np.ndarray): Demanda de cada cliente.
        max_vehicle_capacity (np.ndarray): Capacidades máximas dos veículos.
        max_vehicle_time (np.ndarray): Tempos máximos de operação dos veículos.
    """
    # Matriz de tempo aleatória (inclui o depósito)
    time_matrix = np.random.randint(5, max_time_value, size=(num_customers + 1, num_customers + 1))
    np.fill_diagonal(time_matrix, 0)  # O tempo de um ponto para ele mesmo é zero

    # Demanda aleatória para cada cliente (depósito tem demanda 0)
    demand = np.random.randint(1, max_demand, size=num_customers + 1)
    demand[0] = 0  # Depósito

    # Capacidade máxima de cada veículo
    max_vehicle_capacity = np.random.randint(10, max_capacity, size=num_vehicles)

    # Tempo máximo que cada veículo pode operar
    max_vehicle_time = np.random.randint(50, max_time_limit, size=num_vehicles)

    return time_matrix, demand, max_vehicle_capacity, max_vehicle_time
