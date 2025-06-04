import numpy as np

# Dam break on a wet domain without friction
# https://arxiv.org/pdf/1110.0288

def solve_polynomial_numerically(g_val, h_val, r_val, ht_val):
    """
    Calculates the numerical solutions for c_m when the polynomial equals zero.

    Args:
        g_val (float): Numerical value for g.
        h_val (float): Numerical value for h.
        r_val (float): Numerical value for r.
        ht_val (float): Numerical value for h_t.

    Returns:
        numpy.ndarray: An array of numerical roots for c_m.
                      Returns None if an error occurs (e.g., sqrt of negative number).
    """
    try:
        # Calculate coefficients based on the expanded polynomial:
        # (1 - 8ghr)c_m^4 + 16ghr sqrt(gh_t) c_m^3 - 8g^2h_t h r c_m^2 - g^2h^2r^2 = 0

        coeff_cm4 = (1 - 8 * g_val * h_val * r_val)
        coeff_cm3 = 16 * g_val * h_val * r_val * np.sqrt(g_val * ht_val)
        coeff_cm2 = -8 * (g_val**2) * ht_val * h_val * r_val
        coeff_cm1 = 0  # There is no c_m^1 term in the expanded polynomial
        coeff_const = -(g_val**2) * (h_val**2) * (r_val**2)

        # Create a list of coefficients in descending order of power
        # [coeff_x^n, coeff_x^(n-1), ..., coeff_x^1, coeff_x^0]
        coefficients = [coeff_cm4, coeff_cm3, coeff_cm2, coeff_cm1, coeff_const]

        print(f"\nNumerical Coefficients for the polynomial:")
        print(f"c_m^4: {coeff_cm4}")
        print(f"c_m^3: {coeff_cm3}")
        print(f"c_m^2: {coeff_cm2}")
        print(f"c_m^1: {coeff_cm1}")
        print(f"c_m^0: {coeff_const}")

        # Use numpy.roots to find the roots of the polynomial
        numerical_solutions = np.roots(coefficients)
        return numerical_solutions
    except ValueError as ve:
        print(f"Error calculating numerical solutions: {ve}")
        print("This might happen if, for example, g * h_t is negative, leading to sqrt of a negative number.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during numerical solving: {e}")
        return None

def dambreak_on_wet_no_friction_analytical(T=6, L=10, hl=0.005, hr=0.001, x0=5):
    # Specifically designed to test SWE solver only
    # SWASHES

    xat = lambda t: x0 - t*np.sqrt(hl)
    xbt = lambda t: x0 + t*(2*np.sqrt(hl)-3*cm)


def dambreak_on_wet_no_friction_builder(T=6, L=10, hl=0.005, hr=0.001, dx=0.05):
    # Specifically designed to test SWE solver only
    # SWASHES
    a = 0

if __name__ == "__main__":

    print("--- Numerical Solution Example ---")
    # Example numerical values (you can change these)
    g_example = 9.81  # Acceleration due to gravity
    h_example = 5.0   # Height
    r_example = 0.5   # Radius
    ht_example = 2.0  # Another height parameter

    print(f"Using example values: g={g_example}, h={h_example}, r={r_example}, h_t={ht_example}")
    numerical_solutions_cm = solve_polynomial_numerically(g_example, h_example, r_example, ht_example)

    if numerical_solutions_cm is not None:
        print(f"\nNumerical Solutions for c_m:\n{numerical_solutions_cm}")
        # You might want to filter for real solutions depending on the context
        real_solutions = [sol.real for sol in numerical_solutions_cm if np.isclose(sol.imag, 0)]
        print(f"Real Solutions for c_m (if any):\n{real_solutions}")





    




