import numpy as np


def kalman_filter_two_frames(prev_position, curr_observation):
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
    xPred = A @ prev_position

    # Predicted covariance (assuming initial covariance as zero for simplicity)
    covPred = A @ np.zeros((len(prev_position), len(prev_position))) @ A.T + Q

    # Kalman gain
    S = C @ covPred @ C.T + R
    klm_gain = np.linalg.inv(S) @ (C @ covPred)

    # Update step
    xHat = xPred + klm_gain @ (curr_observation - C @ xPred)

    return [int(i) for i in xHat]

# Example usage
prev_position = np.array([10, 5])  # Previous frame position, in our situation, the prediction of the current frame from the previous frame
curr_observation = np.array([12, 6])  # Current frame observed position


# Estimate current position
estimated_position = kalman_filter_two_frames(prev_position, curr_observation)
print("Estimated Position:", estimated_position)


