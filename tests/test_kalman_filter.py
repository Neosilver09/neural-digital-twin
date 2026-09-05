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
