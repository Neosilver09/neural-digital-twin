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
