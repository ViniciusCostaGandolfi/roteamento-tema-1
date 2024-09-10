
import numpy as np
from sklearn.cluster import KMeans

from ...generate_data import generate_random_vrp_data
from ...genetic_algorithm_vrp.v1.cleark_weight import clarke_wright


def initialize_population(num_individuals, num_customers, num_vehicles):
    population = []

    for _ in range(num_individuals):
        time_matrix, demand, max_vehicle_capacity, max_vehicle_time = generate_random_vrp_data(num_customers, num_vehicles)

        individual_routes = clarke_wright(time_matrix, demand, max_vehicle_capacity, max_vehicle_time)
        population.append(individual_routes)

    return population


# def initialize_population(points: np.ndarray, size: int, volumes: np.ndarray, max_volume: float):
#     num_clusters = int(np.ceil(np.sum(volumes) / max_volume))

#     population = np.empty((size, points.shape[0] - 1), dtype=np.int32)
#     kmeans_population_size = size // 2
    
    

#     for i in range(kmeans_population_size):
#         kmeans = KMeans(n_clusters=num_clusters, random_state=i, n_init=1)
#         labels = kmeans.fit_predict(points[1:])

#         routes = [[] for _ in range(num_clusters)]

#         for j, label in enumerate(labels):
#             routes[label].append(j + 1)

#         individual = np.concatenate([np.array(route) for route in routes])
#         population[i] = individual

#     return population


