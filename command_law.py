import numpy as np


def command_law(T, a, t, xf):
    # Calculate intermediate values
    t2 = 1.0 / T
    t3 = t * t2 * np.pi * 2.0
    t4 = t * t2 * np.pi * 4.0
    t5 = t * t2 * np.pi * 6.0
    t9 = t * t2 * np.pi * 8.0
    t6 = np.sin(t3)
    t7 = np.sin(t4)
    t8 = np.sin(t5)
    t10 = np.sin(t9)

    # Calculate Displacement
    Displacement = t2 * xf * (
                t - (T * (a[0] * t6 + (a[1] * t7) / 2.0 + (a[2] * t8) / 3.0 + (a[3] * t10) / 4.0)) / (np.pi * 2.0))

    # Calculate Velocity
    Velocity = -t2 * xf * (a[0] * np.cos(t3) + a[1] * np.cos(t4) + a[2] * np.cos(t5) + a[3] * np.cos(t9) - 1.0)

    # Calculate Acceleration
    Acceleration = t2 * xf * (
                a[0] * t2 * t6 * np.pi * 2.0 + a[1] * t2 * t7 * np.pi * 4.0 + a[2] * t2 * t8 * np.pi * 6.0 + a[
            3] * t2 * t10 * np.pi * 8.0)

    return Displacement, Velocity, Acceleration
