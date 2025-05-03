import numpy as np

EPSILON = 1e-6
DISTANCE_TO_STATIC_OBSTACLE_THRESHOLD = 10

# Vehicle parameters
L = 2.8        # Wheelbase, e.g. 2.8 m for a typical car
d_safe = 2.0       # Minimum safe distance from obstacles (meters)  
d_tolerance = 1.5 # 5.0

# MPC horizon
N = 10              # Number of steps in the prediction horizon
dt = 0.1          # Timestep (seconds)


# Vehicle characteristics (from CARLA simulator, see get_physics_control())
MAX_CONTROL_WHEEL_ANGLE = 70 / 180 * np.pi               # 1.22 Radian
MAX_CONTROL_ACCELERATION = 10                            # m/s^2
MAX_CONTROL_BRAKING = -4.1                               # m/s^2
MIN_SPEED = 10 # we do not use this

TERMINAL_COST_COEF = 20.0
FINE_STEER_COEF = 0.0 #1.0
FINE_ACC_COEF = 0.0 #0.5
FINE_STEER_DOT_COEF = 100.0 #20.0
FINE_ACC_DOT_COEF = 0.0 #5.0
SOC_COST_COEF = 0.0 # 1000.0
DOC_COST_COEF = 50.0 #1000.0
SOR_COST_COEF = 0.0 #200.0

