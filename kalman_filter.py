"""
Kalman Filter Implementation for 2D Object Tracking
Author: AB Quader
Course: Recursive State Estimation for Dynamic Systems - LUH
"""

import numpy as np


class KalmanFilter:
    """
    Linear Kalman Filter for 2D position and velocity tracking.
    
    State vector: [x, y, vx, vy] — position and velocity in 2D
    Measurement:  [x, y]         — we only observe position
    """

    def __init__(self, dt=0.1, process_noise=1.0, measurement_noise=2.0):
        """
        Initialize Kalman Filter.
        
        Args:
            dt: Time step between measurements (seconds)
            process_noise: How much we trust the motion model
            measurement_noise: How noisy our sensor is
        """
        self.dt = dt

        # --- State transition matrix F ---
        # Describes how state evolves: x_new = F * x_old
        # [x]   [1 0 dt 0 ] [x ]
        # [y] = [0 1 0  dt] [y ]
        # [vx]  [0 0 1  0 ] [vx]
        # [vy]  [0 0 0  1 ] [vy]
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # --- Measurement matrix H ---
        # We only measure position (x, y), not velocity
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # --- Process noise covariance Q ---
        # Uncertainty in our motion model
        self.Q = np.eye(4) * process_noise

        # --- Measurement noise covariance R ---
        # Uncertainty in our sensor
        self.R = np.eye(2) * measurement_noise

        # --- Initial state and covariance ---
        self.x = np.zeros(4)      # [x, y, vx, vy]
        self.P = np.eye(4) * 1.0  # Initial uncertainty

    def initialize(self, x0, y0):
        """Set initial position."""
        self.x = np.array([x0, y0, 0.0, 0.0])
        self.P = np.eye(4) * 1.0

    def predict(self):
        """
        Prediction step — project state forward using motion model.
        
        Returns:
            Predicted state [x, y, vx, vy]
        """
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x.copy()

    def update(self, measurement):
        """
        Update step — correct prediction using new measurement.
        
        Args:
            measurement: [x, y] observed position
            
        Returns:
            Updated state [x, y, vx, vy]
        """
        z = np.array(measurement)

        # Innovation — difference between measurement and prediction
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain — how much to trust measurement vs prediction
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

        return self.x.copy()

    def get_position(self):
        """Return current estimated position."""
        return self.x[0], self.x[1]

    def get_velocity(self):
        """Return current estimated velocity."""
        return self.x[2], self.x[3]
