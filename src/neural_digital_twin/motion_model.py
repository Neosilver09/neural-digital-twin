import numpy as np


class LinearMotionModel:
    def __init__(self, F, H, Q, R, x0, rng):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.rng = rng

    def step(self):
        n = self.F.shape[0]
        m = self.H.shape[0]

        process_noise = self.rng.multivariate_normal(np.zeros(n), self.Q).reshape(-1, 1)
        self.x = self.F @ self.x + process_noise

        measurement_noise = self.rng.multivariate_normal(np.zeros(m), self.R).reshape(-1, 1)
        measurement = self.H @ self.x + measurement_noise

        return self.x, measurement

    def simulate(self, n_steps):
        true_states = []
        measurements = []
        for _ in range(n_steps):
            true_state, measurement = self.step()
            true_states.append(true_state)
            measurements.append(measurement)
        return np.stack(true_states), np.stack(measurements)
