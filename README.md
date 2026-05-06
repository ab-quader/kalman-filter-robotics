# Kalman Filter for 2D Robotics Tracking

Implementation of a **Linear Kalman Filter** and **Extended Kalman Filter** 
for 2D object tracking — built from scratch in Python.

This project is part of my learning journey in the 
*Recursive State Estimation for Dynamic Systems* course at 
Leibniz University Hannover (LUH).

## What It Does
- Tracks a moving object in 2D space
- Fuses noisy sensor measurements with a motion model
- Visualizes true trajectory vs noisy measurements vs KF estimate
- Quantifies the improvement in tracking accuracy

## Results
![Tracking Results](<img width="2540" height="1390" alt="image" src="https://github.com/user-attachments/assets/52bd083a-6499-4e9f-9c2f-62d16c98809f" />
)

## Installation
```bash
git clone https://github.com/ab-quader/kalman-filter-robotics.git
cd kalman-filter-robotics
pip install -r requirements.txt
```

## Usage
```bash
python demo.py
```

## Key Concepts Implemented
- State transition model (constant velocity)
- Kalman gain computation
- Predict → Update cycle
- Error covariance propagation

## What I Learned
- How the Kalman filter balances trust between model and measurement
- Effect of process noise vs measurement noise on filter performance
- Why the innovation covariance S matters

## Next Steps
- [ ] Extended Kalman Filter (EKF) for nonlinear systems
- [ ] Particle Filter implementation
- [ ] Apply to ROS 2 robot simulation

## Author
A. Quader — M.Sc. AI-driven Mechatronics & Robotics, LUH Hannover
