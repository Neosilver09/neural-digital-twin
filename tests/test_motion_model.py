import numpy as np

from neural_digital_twin.motion_model import LinearMotionModel


def test_constructor_stores_initial_true_state():
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.01, 0.0], [0.0, 0.01]])
    R = np.array([[0.5]])
    x0 = np.array([[0.0], [2.0]])

    model = LinearMotionModel(F=F, H=H, Q=Q, R=R, x0=x0, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(model.x, x0)


def test_step_is_deterministic_when_process_and_measurement_noise_are_zero():
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.zeros((2, 2))
    R = np.zeros((1, 1))
    x0 = np.array([[0.0], [2.0]])

    model = LinearMotionModel(F=F, H=H, Q=Q, R=R, x0=x0, rng=np.random.default_rng(123))
    true_state, measurement = model.step()

    # x = F @ x0 exactly, since Q = 0
    np.testing.assert_allclose(true_state, np.array([[2.0], [2.0]]), atol=1e-8)
    # z = H @ x exactly, since R = 0
    np.testing.assert_allclose(measurement, np.array([[2.0]]), atol=1e-8)


def test_step_applies_genuine_process_and_measurement_noise():
    # F = identity isolates noise from dynamics: with no noise, x would stay
    # exactly x0 and z would stay exactly H @ x0. Any deviation must be noise.
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.5, 0.0], [0.0, 0.5]])
    R = np.array([[0.1]])
    x0 = np.array([[1.0], [1.0]])

    model = LinearMotionModel(F=F, H=H, Q=Q, R=R, x0=x0, rng=np.random.default_rng(7))
    true_state, measurement = model.step()

    assert not np.allclose(true_state, x0)
    assert not np.allclose(measurement, H @ x0)


def test_simulate_matches_repeated_step_calls_with_identically_seeded_rng():
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.05, 0.0], [0.0, 0.05]])
    R = np.array([[0.2]])
    x0 = np.array([[0.0], [1.0]])
    n_steps = 5

    model_a = LinearMotionModel(F=F, H=H, Q=Q, R=R, x0=x0, rng=np.random.default_rng(99))
    true_states, measurements = model_a.simulate(n_steps)

    assert true_states.shape == (n_steps, 2, 1)
    assert measurements.shape == (n_steps, 1, 1)

    model_b = LinearMotionModel(F=F, H=H, Q=Q, R=R, x0=x0, rng=np.random.default_rng(99))
    manual_true_states = []
    manual_measurements = []
    for _ in range(n_steps):
        true_state, measurement = model_b.step()
        manual_true_states.append(true_state)
        manual_measurements.append(measurement)

    np.testing.assert_allclose(true_states, np.stack(manual_true_states))
    np.testing.assert_allclose(measurements, np.stack(manual_measurements))


def test_measurement_noise_variance_matches_theoretical_R_over_many_steps():
    # F = I, Q = 0: the true state never moves, so every fluctuation in the
    # measurements is attributable entirely to R. Over many steps, the
    # sample variance of the measurements should converge to R.
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.zeros((2, 2))
    R = np.array([[0.8]])
    x0 = np.array([[3.0], [0.0]])
    n_steps = 5000

    model = LinearMotionModel(F=F, H=H, Q=Q, R=R, x0=x0, rng=np.random.default_rng(2024))
    _, measurements = model.simulate(n_steps)

    sample_variance = np.var(measurements.reshape(-1), ddof=1)
    relative_error = abs(sample_variance - R[0, 0]) / R[0, 0]

    # Theoretical relative std. error of sample variance for n i.i.d. Gaussian
    # draws is sqrt(2 / (n-1)) ~= 2% here. 8% is ~4 std. devs -- tight enough
    # to catch a real miscalibration, generous enough to not be flaky.
    assert relative_error < 0.08
