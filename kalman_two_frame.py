import numpy as np


def kf_one_step(prev_position, curr_observation, covHat):
    """
    Kalman Filter for two-frame position estimation.
    
    :param prev_position: Position in the previous frame (numpy array).
    :param curr_observation: Observed position in the current frame (numpy array).
    :return: Estimated position in the current frame (numpy array).
    """

    Q = 1e-3 * np.eye(2)  # Process noise covariance
    R = 0.001 * np.eye(2)  # Observation noise covariance
    # State transition matrix (assuming simple forward movement)
    A = np.eye(len(prev_position))

    # Observation matrix
    C = np.eye(len(prev_position))

    # Predicted state (assuming no control input)

    xPred = A @ prev_position  # prev_position should be xHat from the previous frame

    # Predicted covariance (assuming initial covariance as zero for simplicity)
    covPred = A @ covHat @ A.T + Q  # covHat is from the previous frame

    # Kalman gain
    S = C @ covPred @ C.T + R  # Biased norm
    # klm_gain = np.linalg.inv(S) @ (C @ covPred)
    klm_gain = covPred @ C.T @ np.linalg.inv(S)

    # Update step
    xHat = xPred + klm_gain @ (curr_observation - C @ xPred)
    covHat = covPred - klm_gain @ C @ covPred

    return [int(i) for i in xHat], covHat
    # return xHat, covHat


if __name__ == "__main__":
    # Example usage
    prev_position = np.array([0, 0])  # Previous frame position, in our situation, the prediction of the current frame from the previous frame
    curr_observation = np.array([14, 5])  # Current frame observed position

    # Estimate current position
    estimated_position, covHat = kf_one_step(prev_position, curr_observation, np.zeros((2, 2)))
    print("Estimated Position:", estimated_position)


