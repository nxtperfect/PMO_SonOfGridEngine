import numpy as np


class Particle:
    def __init__(self, dimensions, bounds):
        # Initialize particle position
        self.position = np.array(
            [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(dimensions)]
        )

        # Initialize velocity
        self.velocity = np.random.uniform(-1, 1, dimensions)

        # Personal best
        self.best_position = np.copy(self.position)
        self.best_score = float("inf")


class PSO:
    def __init__(
        self,
        objective_function,
        dimensions,
        bounds,
        num_particles=30,
        max_iterations=100,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
    ):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iterations = max_iterations

        # PSO coefficients
        self.w = inertia  # inertia weight
        self.c1 = cognitive  # cognitive coefficient
        self.c2 = social  # social coefficient

        # Create swarm
        self.swarm = [Particle(dimensions, bounds) for _ in range(num_particles)]

        # Global best
        self.global_best_position = np.zeros(dimensions)
        self.global_best_score = float("inf")

    def optimize(self):
        for iteration in range(self.max_iterations):

            for particle in self.swarm:

                # Evaluate fitness
                fitness = self.objective_function(particle.position)

                # Update personal best
                if fitness < particle.best_score:
                    particle.best_score = fitness
                    particle.best_position = np.copy(particle.position)

                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = np.copy(particle.position)

            # Update velocities and positions
            for particle in self.swarm:

                r1 = np.random.rand(self.dimensions)
                r2 = np.random.rand(self.dimensions)

                cognitive_velocity = (
                    self.c1 * r1 * (particle.best_position - particle.position)
                )

                social_velocity = (
                    self.c2 * r2 * (self.global_best_position - particle.position)
                )

                # Velocity update
                particle.velocity = (
                    self.w * particle.velocity + cognitive_velocity + social_velocity
                )

                # Position update
                particle.position += particle.velocity

                # Apply bounds
                for i in range(self.dimensions):
                    particle.position[i] = np.clip(
                        particle.position[i], self.bounds[i][0], self.bounds[i][1]
                    )

            print(
                f"Iteration {iteration + 1}/{self.max_iterations}, "
                f"Best Score = {self.global_best_score:.24f}"
            )

        return self.global_best_position, self.global_best_score


# Example objective function
# Sphere function: minimum at x = [0,0,...]
def sphere_function(x):
    return np.sum(x**2)


if __name__ == "__main__":

    dimensions = 2

    bounds = [(-10, 10), (-10, 10)]

    pso = PSO(
        objective_function=sphere_function,
        dimensions=dimensions,
        bounds=bounds,
        num_particles=30,
        max_iterations=100,
    )

    best_position, best_score = pso.optimize()

    print("\nOptimization Finished")
    print(f"Best Position: {best_position}")
    print(f"Best Score: {best_score:.24f}")
