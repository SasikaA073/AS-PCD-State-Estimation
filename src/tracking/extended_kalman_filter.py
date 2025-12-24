import numpy as np

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for 3D tracking.
    State Model: Constant Velocity (Linear).
    Measurement Model: Radar/Lidar style (Range, Azimuth, Elevation) - Non-Linear.
    
    State Vector: [x, y, z, vx, vy, vz] (6x1)
    Measurement Vector: [rho, theta, phi] (3x1) (Range, Azimuth, Elevation)
    """
    def __init__(self, dt=1.0, process_noise_std=0.1, measurement_noise_std=0.1):
        self.dt = dt
        self.state_dim = 6
        self.meas_dim = 3
        
        # 1. State Vector
        self.x = np.zeros((self.state_dim, 1))
        
        # 2. State Covariance
        self.P = np.eye(self.state_dim) * 100.0
        
        # 3. State Transition Matrix F (Linear CV Model)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 4. Process Noise Q (Same as Linear KF)
        self.Q = np.eye(self.state_dim) * (process_noise_std ** 2)
        
        # 5. Measurement Noise R
        self.R = np.eye(self.meas_dim) * (measurement_noise_std ** 2)

    def predict(self):
        """
        Predict step is identical to Linear KF because our process model is linear.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
        
    def h(self, x):
        """
        Non-linear Measurement Function h(x).
        Converts Cartesian state (x, y, z) to Spherical (rho, theta, phi).
        """
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]
        
        rho = np.sqrt(px**2 + py**2 + pz**2)
        
        # Avoid division by zero
        if rho < 1e-4:
            rho = 1e-4
            
        theta = np.arctan2(py, px) # Azimuth
        phi = np.arcsin(pz / rho)  # Elevation (assuming z is up)
        
        return np.array([[rho], [theta], [phi]])

    def calculate_jacobian(self, x):
        """
        Calculates the Jacobian matrix Hj of the measurement function h(x).
        """
        px, py, pz = x[0, 0], x[1, 0], x[2, 0]
        
        c1 = px**2 + py**2 + pz**2
        c2 = np.sqrt(c1)
        c3 = px**2 + py**2
        c4 = np.sqrt(c3)
        
        # Check division by zero
        if np.abs(c1) < 1e-4 or np.abs(c3) < 1e-4:
            # Return zeros or previous valid Jacobian if extremely close to origin
            return np.zeros((3, 6))

        # Jacobian elements
        # Row 1: d(rho)/d...
        # rho = sqrt(x^2 + y^2 + z^2)
        dr_dx = px / c2
        dr_dy = py / c2
        dr_dz = pz / c2
        
        # Row 2: d(theta)/d... (Azimuth)
        # theta = atan2(y, x) -> deriv is [-y/(x^2+y^2), x/(x^2+y^2), 0]
        # c3 = x^2 + y^2
        dt_dx = -py / c3
        dt_dy = px / c3
        dt_dz = 0
        
        # Row 3: d(phi)/d... (Elevation)
        # phi = asin(z / rho)
        # d(phi)/dx = -xz / (rho^2 * sqrt(x^2+y^2))
        # d(phi)/dy = -yz / (rho^2 * sqrt(x^2+y^2))
        # d(phi)/dz = sqrt(x^2+y^2) / rho^2
        
        # Term denominators
        # rho^2 * sqrt(x^2+y^2) = (x^2+y^2+z^2) * sqrt(x^2+y^2) = c1 * c4
        
        denom = c1 * c4
        if abs(denom) < 1e-4: denom = 1e-4 # Guard

        dp_dx = -(px * pz) / denom
        dp_dy = -(py * pz) / denom
        dp_dz = c4 / c1

        Hj = np.array([
            [dr_dx, dr_dy, dr_dz, 0, 0, 0],
            [dt_dx, dt_dy, dt_dz, 0, 0, 0],
            [dp_dx, dp_dy, dp_dz, 0, 0, 0]
        ])
        
        return Hj

    def update(self, z):
        """
        EKF Update Step.
        Args:
            z: Measurement vector [rho, theta, phi]
        """
        z = np.array(z).reshape((3, 1))
        
        # 1. Linearize Measurement Model at current state prediction
        Hj = self.calculate_jacobian(self.x)
        
        # 2. Calculate Innovation (Measurement Residual)
        # y = z - h(x_pred)
        z_pred = self.h(self.x)
        y = z - z_pred
        
        # 3. Normalize Angles in y (important!)
        # Azimuth theta should be in [-pi, pi]
        while y[1, 0] > np.pi: y[1, 0] -= 2. * np.pi
        while y[1, 0] < -np.pi: y[1, 0] += 2. * np.pi
        # Elevation phi norm (optional depending on convention, usually [-pi/2, pi/2])
        
        # 4. Standard Kalman Update using Hj instead of H
        S = Hj @ self.P @ Hj.T + self.R
        K = self.P @ Hj.T @ np.linalg.inv(S)
        
        self.x = self.x + (K @ y)
        I = np.eye(self.state_dim)
        self.P = (I - (K @ Hj)) @ self.P
        
        return self.x

    def get_position(self):
        """Returns current [x, y, z] estimate."""
        return self.x[:3].flatten()
