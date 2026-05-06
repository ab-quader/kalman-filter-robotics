"""
Demo: Kalman Filter tracking a moving object in 2D
Author: AB Quader
"""

import numpy as np
from kalman_filter import KalmanFilter
from visualization import plot_tracking_results


def generate_trajectory(n_steps=100, dt=0.1):
    """
    Generate a smooth curved trajectory (ground truth).
    Simulates an object moving in a curved path.
    """
    t = np.linspace(0, 2 * np.pi, n_steps)
    
    # Circular-ish path with some variation
    x = 10 * np.cos(t) + 0.5 * t
    y = 10 * np.sin(t) + 0.3 * t
    
    return list(zip(x, y))


def add_noise(positions, noise_std=1.5):
    """Add Gaussian noise to simulate sensor measurements."""
    noisy = []
    for x, y in positions:
        nx = x + np.random.normal(0, noise_std)
        ny = y + np.random.normal(0, noise_std)
        noisy.append((nx, ny))
    return noisy


def main():
    print("🤖 Kalman Filter Demo — AB Quader")
    print("=" * 40)
    
    # --- Setup ---
    np.random.seed(42)
    dt = 0.1
    n_steps = 100

    # Generate ground truth trajectory
    true_positions = generate_trajectory(n_steps, dt)
    
    # Simulate noisy sensor measurements
    noisy_measurements = add_noise(true_positions, noise_std=1.5)
    
    # --- Run Kalman Filter ---
    kf = KalmanFilter(dt=dt, process_noise=0.5, measurement_noise=2.0)
    kf.initialize(*true_positions[0])
    
    kf_estimates = []
    
    print(f"Running Kalman Filter on {n_steps} timesteps...")
    
    for i, measurement in enumerate(noisy_measurements):
        # Predict
        kf.predict()
        
        # Update with measurement
        kf.update(measurement)
        
        # Store estimate
        pos = kf.get_position()
        kf_estimates.append(pos)

    print("✅ Filtering complete!")
    
    # --- Visualize ---
    plot_tracking_results(
        true_positions,
        noisy_measurements,
        kf_estimates,
        title="Kalman Filter: 2D Object Tracking\n(LUH — Recursive State Estimation)"
    )


if __name__ == "__main__":
    main()
