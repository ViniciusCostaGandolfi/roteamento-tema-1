


from .v1.inicial_populaton import generate_population


num_customers = 10
num_vehicles = 3
num_individuals = 100

population = generate_population(num_individuals, num_customers, num_vehicles)

print(f"\nTotal de indivíduos gerados: {len(population)}")
for idx, individual in enumerate(population[:3], 1):
    print(f"\nIndivíduo {idx}:")
    for route in individual:
        print(f"Rota: {route}")