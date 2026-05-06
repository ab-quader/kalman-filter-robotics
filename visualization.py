"""
Visualization for Kalman Filter tracking results.
Author: AB Quader
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_tracking_results(true_positions, noisy_measurements, 
                           kf_estimates, title="Kalman Filter Tracking"):
    """
    Plot true trajectory, noisy measurements, and KF estimates.
    
    Args:
        true_positions:    list of (x, y) — ground truth
        noisy_measurements: list of (x, y) — raw sensor data
        kf_estimates:      list of (x, y) — Kalman filter output
        title:             plot title
    """
    true_positions = np.array(true_positions)
    noisy_measurements = np.array(noisy_measurements)
    kf_estimates = np.array(kf_estimates)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # --- Plot 1: 2D Trajectory ---
    ax1 = axes[0]
    ax1.plot(true_positions[:, 0], true_positions[:, 1],
             'g-', linewidth=2, label='True Trajectory', zorder=3)
    ax1.scatter(noisy_measurements[:, 0], noisy_measurements[:, 1],
                c='red', s=15, alpha=0.5, label='Noisy Measurements', zorder=2)
    ax1.plot(kf_estimates[:, 0], kf_estimates[:, 1],
             'b-', linewidth=2, label='Kalman Filter Estimate', zorder=4)

    # Mark start and end
    ax1.scatter(*true_positions[0], c='green', s=100, 
                marker='*', zorder=5, label='Start')
    ax1.scatter(*true_positions[-1], c='black', s=100, 
                marker='X', zorder=5, label='End')

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # --- Plot 2: Error over time ---
    ax2 = axes[1]
    
    measurement_error = np.sqrt(
        (noisy_measurements[:, 0] - true_positions[:, 0])**2 +
        (noisy_measurements[:, 1] - true_positions[:, 1])**2
    )
    kf_error = np.sqrt(
        (kf_estimates[:, 0] - true_positions[:, 0])**2 +
        (kf_estimates[:, 1] - true_positions[:, 1])**2
    )
    
    timesteps = np.arange(len(true_positions))
    ax2.plot(timesteps, measurement_error, 'r-', 
             alpha=0.7, label=f'Measurement Error (avg: {measurement_error.mean():.2f}m)')
    ax2.plot(timesteps, kf_error, 'b-', 
             linewidth=2, label=f'KF Error (avg: {kf_error.mean():.2f}m)')
    ax2.fill_between(timesteps, measurement_error, kf_error, 
                     alpha=0.2, color='green', label='Improvement')

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Tracking Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tracking_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n📊 Results saved to tracking_results.png")
    print(f"   Average measurement error: {measurement_error.mean():.3f} m")
    print(f"   Average KF estimate error: {kf_error.mean():.3f} m")
    print(f"   Improvement: {((measurement_error.mean() - kf_error.mean()) / measurement_error.mean() * 100):.1f}%")
