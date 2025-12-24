import numpy as np

class KalmanFilter:
    """
    A standard Linear Kalman Filter for 3D tracking.
    Assumes a Constant Velocity (CV) model.
    
    State Vector: [x, y, z, vx, vy, vz] (6x1)
    Measurement Vector: [x, y, z] (3x1)
    """
    def __init__(self, dt=1.0, process_noise_std=0.1, measurement_noise_std=0.1):
        """
        Args:
            dt (float): Time step between updates.
            process_noise_std (float): Standard deviation of process noise (Q).
            measurement_noise_std (float): Standard deviation of measurement noise (R).
        """
        self.dt = dt
        self.state_dim = 6
        self.meas_dim = 3
        
        # 1. State Vector [x, y, z, vx, vy, vz]
        self.x = np.zeros((self.state_dim, 1))
        
        # 2. State Covariance Matrix P
        # Initial high uncertainty
        self.P = np.eye(self.state_dim) * 100.0
        
        # 3. State Transition Matrix F (Constant Velocity Model)
        # x_new = x + vx * dt
        # v_new = v
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 4. Measurement Matrix H
        # We only observe positions [x, y, z]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # 5. Process Noise Covariance Q
        # Models uncertainty in the physics (e.g. slight acceleration changes)
        # Simplified Q matrix
        q_pos = (dt**4)/4 * (process_noise_std**2)
        q_vel = (dt**2) * (process_noise_std**2)
        # Ideally Q should be more coupled, but diagonal approximation is often sufficient for simple caching
        # Here is a diagonal-ish approximation for simplicity
        self.Q = np.eye(self.state_dim) * (process_noise_std ** 2)
        
        # 6. Measurement Noise Covariance R
        # Models sensor noise
        self.R = np.eye(self.meas_dim) * (measurement_noise_std ** 2)

    def predict(self):
        """
        Predict the next state based on the process model.
        x = Fx
        P = FPF' + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        Update the state estimate with a new measurement.
        Args:
            z (np.array): Measurement vector [x, y, z] (3x1) or (3,)
        """
        z = np.array(z).reshape((self.meas_dim, 1))
        
        # Innovation (residual)
        y = z - (self.H @ self.x)
        
        # Innovation Covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update State
        self.x = self.x + (K @ y)
        
        # Update Covariance
        I = np.eye(self.state_dim)
        self.P = (I - (K @ self.H)) @ self.P
        
        return self.x

    def get_position(self):
        """Returns current [x, y, z] estimate."""
        return self.x[:3].flatten()
