import casadi as ca
from carla_api.mpc.config import L


def bicycle_model(state, control, dt, L=L):

    # c1 = steering angle velocity of front wheels
    # c2 = acceleration

    x, y, v, yaw = state[0], state[1], state[2], state[3]
    delta, acc = control[0], control[1]
    
    x_next = x + v*ca.cos(yaw)*dt      
    y_next = y + v*ca.sin(yaw)*dt
    yaw_next = yaw + (v/L)*ca.tan(delta)*dt
    v_next = v + acc*dt

    return ca.vertcat(x_next, y_next, v_next, yaw_next)
