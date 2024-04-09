from sympy import symbols, sin, cos, diff, pi, lambdify


def motion_description():
    # Define symbols
    n = 4
    t, T, xf = symbols('t T xf')
    a = symbols(f'a:{n}')  # Creates a tuple of symbols a_0, a_1, ..., a_(n-1)

    # Duhamel integral calculation
    Summation_Dis = [(a[i]/(i+1))*sin((2*pi*(i+1)*t)/T) for i in range(n)]
    Summation_Vel = [a[i]*cos((2*pi*(i+1)*t)/T) for i in range(n)]

    # Calculate Displacement, Velocity, Acceleration, and Jerk
    Dis = (xf/T)*(t-(T/(2*pi))*sum(Summation_Dis))
    V = (xf/T)*(1-sum(Summation_Vel))
    Acc = diff(V, t)
    J = diff(Acc, t)

    # Convert to callable functions using lambdify
    Dis_function = lambdify((t, T, xf, *a), Dis, 'numpy')
    V_function = lambdify((t, T, xf, *a), V, 'numpy')
    Acc_function = lambdify((t, T, xf, *a), Acc, 'numpy')
    J_function = lambdify((t, T, xf, *a), J, 'numpy')

    return Dis_function, V_function, Acc_function, J_function
