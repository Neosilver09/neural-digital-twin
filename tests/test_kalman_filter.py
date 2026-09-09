import numpy as np

from neural_digital_twin.kalman_filter import KalmanFilter


def test_constructor_stores_initial_state_and_covariance():
    F = np.array([[1.5]])
    H = np.array([[1.0]])
    Q = np.array([[0.5]])
    R = np.array([[1.0]])
    x0 = np.array([[5.0]])
    P0 = np.array([[3.0]])

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

    np.testing.assert_array_equal(kf.x, x0)
    np.testing.assert_array_equal(kf.P, P0)


def test_predict_updates_state_and_covariance_using_transition_model():
    F = np.array([[1.5]])
    H = np.array([[1.0]])
    Q = np.array([[0.5]])
    R = np.array([[1.0]])
    x0 = np.array([[5.0]])
    P0 = np.array([[3.0]])

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)
    kf.predict()

    # x = F @ x0 = 1.5 * 5.0
    np.testing.assert_allclose(kf.x, np.array([[7.5]]))
    # P = F @ P0 @ F.T + Q = 1.5 * 3.0 * 1.5 + 0.5
    np.testing.assert_allclose(kf.P, np.array([[7.25]]))


def test_update_computes_posterior_state_and_covariance_using_measurement():
    F = np.array([[1.5]])
    H = np.array([[1.0]])
    Q = np.array([[0.5]])
    R = np.array([[1.0]])
    # Prior (x, P): the predicted state from the previous test, i.e. the
    # output of predict() just before a new measurement arrives.
    x_prior = np.array([[7.5]])
    P_prior = np.array([[7.25]])

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x_prior, P0=P_prior)
    z = np.array([[10.0]])
    kf.update(z)

    # y = z - H @ x = 10.0 - 7.5
    innovation = 10.0 - 7.5
    # S = H @ P @ H.T + R = 7.25 + 1.0
    innovation_covariance = 7.25 + 1.0
    # K = P @ H.T @ inv(S) = 7.25 / 8.25
    kalman_gain = 7.25 / innovation_covariance
    expected_x = 7.5 + kalman_gain * innovation
    # P = (I - K @ H) @ P = (1 - K) * 7.25
    expected_P = (1 - kalman_gain) * 7.25

    np.testing.assert_allclose(kf.x, np.array([[expected_x]]))
    np.testing.assert_allclose(kf.P, np.array([[expected_P]]))


def test_update_barely_moves_state_when_measurement_noise_is_huge():
    F = np.array([[1.5]])
    H = np.array([[1.0]])
    Q = np.array([[0.5]])
    R = np.array([[1.0e6]])  # sensor is essentially untrustworthy
    x_prior = np.array([[7.5]])
    P_prior = np.array([[7.25]])

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x_prior, P0=P_prior)
    kf.update(np.array([[10.0]]))  # measurement disagrees sharply with prior

    np.testing.assert_allclose(kf.x, x_prior, atol=1e-3)


def test_update_moves_state_close_to_measurement_when_measurement_noise_is_tiny():
    F = np.array([[1.5]])
    H = np.array([[1.0]])
    Q = np.array([[0.5]])
    R = np.array([[1.0e-6]])  # sensor is essentially exact
    x_prior = np.array([[7.5]])
    P_prior = np.array([[7.25]])

    z = np.array([[10.0]])
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x_prior, P0=P_prior)
    kf.update(z)

    np.testing.assert_allclose(kf.x, z, atol=1e-3)


def test_constant_velocity_model_tracks_true_position_and_velocity():
    # State is [position, velocity]^T; only position is measured. This is
    # the first test using genuine (non-1x1) matrices, exercising matrix
    # multiplication and transposition that scalar tests couldn't catch bugs in.
    dt = 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1e-4, 0.0], [0.0, 1e-4]])
    R = np.array([[0.5]])
    x0 = np.array([[0.0], [0.0]])  # no prior belief about position or velocity
    P0 = np.array([[10.0, 0.0], [0.0, 10.0]])  # so start very uncertain

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

    true_velocity = 2.0
    n_steps = 30
    measurement_noise_std = np.sqrt(R[0, 0])
    rng = np.random.default_rng(42)

    for step in range(1, n_steps + 1):
        true_position = true_velocity * step * dt
        z = np.array([[true_position + rng.normal(0.0, measurement_noise_std)]])
        kf.predict()
        kf.update(z)

    final_true_position = true_velocity * n_steps * dt
    estimated_position = kf.x[0, 0]
    estimated_velocity = kf.x[1, 0]

    # Tolerances are a few multiples of the actual residual observed with this
    # seed (~0.18 position, ~0.01 velocity) -- tight enough to catch a
    # regression (e.g. a sign error or an ignored measurement), loose enough
    # to not be flaky if the seed or noise realization changes.
    assert abs(estimated_position - final_true_position) < 0.5
    assert abs(estimated_velocity - true_velocity) < 0.05


def test_repeated_predict_without_update_strictly_grows_covariance():
    # F = identity isolates the property: with no dynamics reshaping the
    # state, repeated predict() should still accumulate uncertainty purely
    # from process noise Q, since no measurement ever arrives to correct it.
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    Q = np.array([[0.1, 0.0], [0.0, 0.2]])
    R = np.array([[1.0]])
    x0 = np.array([[0.0], [0.0]])
    P0 = np.array([[1.0, 0.0], [0.0, 1.0]])

    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

    n_steps = 10
    traces = [np.trace(kf.P)]
    for _ in range(n_steps):
        kf.predict()
        traces.append(np.trace(kf.P))

    # Strict growth at every single step, not just start-vs-end.
    assert all(
        later > earlier for earlier, later in zip(traces, traces[1:], strict=False)
    )

    # F = I means the recursion is exact: P_k = P0 + k * Q.
    expected_final_P = P0 + n_steps * Q
    np.testing.assert_allclose(kf.P, expected_final_P)
