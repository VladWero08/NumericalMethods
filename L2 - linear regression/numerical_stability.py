# 1. xk+1 = {
# 	2 * xk		, xk apartine [0, 0.5)
# 	2 * xk - 1	, xk apartine (0.5, 1]
# }

# a) x0 = 0.1
# x1 = 0.2
# x2 = 0.4
# x3 = 0.8
# x4 = 0.6
# x5 = 0.2
# x6 = 0.4
# x7 = 0.8
# x8 = 0.6
# x9 = 0.2
# x10 = 0.4
# x11 = 0.8
# x12 = 0.6
# ... xk+1 se calculeaza periodic, iar x60 = 0.6

# c) Diferentele apar in momentul in care facem impartirile.

def generate_x(iterations, initial_iterations, x_value):
    if iterations < 0:
        return x_value
    
    print(f"Step {initial_iterations - iterations}: {x_value}")
    
    if 0.0 <= x_value < 0.5:
        return generate_x(iterations - 1, initial_iterations, 2 * x_value)
    else:
        return generate_x(iterations - 1, initial_iterations, 2 * x_value - 1)
    
generate_x(60, 60,  0.1)