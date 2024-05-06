import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
L1, L2 = 1.0, 1.0  # Length of pendulum arms
m1, m2 = 1.0, 1.0  # Masses
g = 9.81  # Acceleration due to gravity


def derivs(t, state):
    """
    Calculates the derivatives of the state variables.
    """
    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
    dydx[1] = (
        m2 * L1 * state[1] * state[1] * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(state[2]) * np.cos(delta)
        + m2 * L2 * state[3] * state[3] * np.sin(delta)
        - (m1 + m2) * g * np.sin(state[0])
    ) / den1

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = (
        -m2 * L2 * state[3] * state[3] * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * g * np.sin(state[0]) * np.cos(delta)
        - (m1 + m2) * L1 * state[1] * state[1] * np.sin(delta)
        - (m1 + m2) * g * np.sin(state[2])
    ) / den2

    return dydx


# Set up initial conditions
theta1 = np.pi / 2
omega1 = 0.0
theta2 = np.pi / 2
omega2 = 0.0

# Create a time array from 0 to 10 seconds with 1000 points (0.01s intervals)
t = np.linspace(0, 10, 1000)

# Bundle initial conditions for the solver
state = [theta1, omega1, theta2, omega2]

# Use scipy's ODE solver to integrate the equations of motion
sol = solve_ivp(derivs, [0, 10], state, t_eval=t)

# Extracting the results
theta1 = sol.y[0]
theta2 = sol.y[2]

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect("equal")
ax.grid()

# Create empty lines for the pendulum arms
(line,) = ax.plot([], [], lw=2)
(line2,) = ax.plot([], [], lw=2)


# Initialize the animation
def init():
    line.set_data([], [])
    line2.set_data([], [])
    return (
        line,
        line2,
    )


# Update function for the animation
def update(frame):
    x1 = L1 * np.sin(theta1[frame])
    y1 = -L1 * np.cos(theta1[frame])

    x2 = x1 + L2 * np.sin(theta2[frame])
    y2 = y1 - L2 * np.cos(theta2[frame])

    line.set_data([0, x1], [0, y1])
    line2.set_data([x1, x2], [y1, y2])
    return (
        line,
        line2,
    )


# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)

plt.show()
