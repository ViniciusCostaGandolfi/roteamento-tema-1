import time
import numpy as np
from numba import jit
import plotly.graph_objects as go
from sklearn.datasets import make_blobs

@jit(nopython=True, cache=True)
def calculate_tour_length(tour, dist_matrix):
    length = 0.0
    for i in range(len(tour) - 1):
        length += dist_matrix[tour[i], tour[i + 1]]
    length += dist_matrix[tour[-1], tour[0]]
    return length

@jit(nopython=True, cache=True)
def half_find_best_2opt_move(tour, dist_matrix):
    n = len(tour)
    best_delta = 0
    best_i, best_j = -1, -1
    for i in range(n - 1):
        distances = np.array([dist_matrix[tour[i], tour[k]] for k in range(n)])
        distances[i] = np.inf  # Impede i de ser o mais próximo
        
        sorted_indices = np.argsort(distances)

        for j in sorted_indices:
            delta = (dist_matrix[tour[i], tour[j % n]] +
                     dist_matrix[tour[(i + 1) % n], tour[(j + 1) % n]] -
                     dist_matrix[tour[i], tour[(i + 1) % n]] -
                     dist_matrix[tour[j % n], tour[(j + 1) % n]])
            delta = round(delta, 3)

            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j
    return best_i, best_j, best_delta

@jit(nopython=True, cache=True)
def find_best_2opt_move(tour, dist_matrix):
    n = len(tour)
    best_delta = 0
    best_i, best_j = -1, -1
    for i in range(n - 1):

        for j in range(i + 2, n + (i > 0)):
            delta = (dist_matrix[tour[i], tour[j % n]] +
                     dist_matrix[tour[(i + 1) % n], tour[(j + 1) % n]] -
                     dist_matrix[tour[i], tour[(i + 1) % n]] -
                     dist_matrix[tour[j % n], tour[(j + 1) % n]])
            delta = round(delta, 3)

            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j
    return best_i, best_j, best_delta

@jit(nopython=True, cache=True)
def apply_2opt_move(tour, i, j):
    new_tour = tour.copy()
    new_tour[i + 1:j + 1] = new_tour[i + 1:j + 1][::-1]
    return new_tour

@jit(nopython=True, cache=True)
def calculate_distance_matrix(points):
    n_points = len(points)
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
            else:
                dist_matrix[i, j] = np.inf
    return dist_matrix

@jit(nopython=True, cache=True)
def lin_kernighan_heuristic(dist_matrix: np.ndarray):
    n = dist_matrix.shape[0]
    tour = np.arange(0, n, dtype=np.int32)
    improved = True
    i = 0
    while improved:

        improved = False
        best_i, best_j, best_delta = find_best_2opt_move(tour, dist_matrix)
        if best_delta < 0:
            tour = apply_2opt_move(tour, best_i, best_j)
            improved = True
        i += 1

    tour_length = calculate_tour_length(tour, dist_matrix)
    return tour, tour_length


n_points = 1000
points, _ = make_blobs(n_samples=n_points, centers=5, cluster_std=2, random_state=42)

# Calcular a matriz de distâncias
dist_matrix = calculate_distance_matrix(points)

# Aplicar o LKH otimizado com raio de busca 5
t = time.time()
tour, tour_length = lin_kernighan_heuristic(dist_matrix)

# Plotar o gráfico
fig = go.Figure()

# Adicionar os pontos ao gráfico
fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', marker=dict(size=8, color='blue')))

# Adicionar as arestas da solução ao gráfico
for i in range(len(tour)):
    start_node = tour[i]
    end_node = tour[(i + 1) % len(tour)]
    fig.add_trace(go.Scatter(x=[points[start_node, 0], points[end_node, 0]],
                             y=[points[start_node, 1], points[end_node, 1]],
                             mode='lines',
                             line=dict(color='red', width=2)))

fig.update_layout(
    title=f'LKH Optimized Tour (Length: {tour_length:.2f}) in {time.time() - t :.2f}s',
    xaxis_title='X',
    yaxis_title='Y',
    showlegend=False,
)

fig.show()