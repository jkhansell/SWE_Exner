import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import functools
import scipy.integrate as integrate

# Dam break on a wet domain without friction
# https://arxiv.org/pdf/1110.0288

g = 9.81
def solve_polynomial_numerically(hl, hr):
    # Fallback to fsolve if polynomial root finding fails or selects an unphysical one
    def equation_for_cm(cm, g, hlt, hrt):

        cr = np.sqrt(g*hrt)
        cl = np.sqrt(g*hlt)

        term1 = -8 * cr**2 * cm**2 * (cl - cm)**2
        term2 = (cm**2 - cr**2)**2 * (cm**2 + cr**2)
        return term1 + term2

    func_to_solve_cm = functools.partial(equation_for_cm, g=g, hlt=hl, hrt=hr)
    initial_guess_cm = (np.sqrt(g * hl) + np.sqrt(g * hr))
    c_m = fsolve(func_to_solve_cm, initial_guess_cm)
    print("Obtained c_m: {}".format(c_m))

    return c_m[0]


def dambreak_on_wet_no_friction_analytical(t, x, L=10, hl=0.005, hr=0.001, x0=5):
    # Specifically designed to test SWE solver only
    # SWASHES

    cm = solve_polynomial_numerically(hl, hr)

    xat = lambda t: x0 - t*np.sqrt(g*hl)
    xbt = lambda t: x0 + t*(2*np.sqrt(g*hl)-3*cm)
    xct = lambda t: x0 + t*(2*cm**2*(np.sqrt(g*hl)-cm))/(cm**2-g*hr)

    h_1 = lambda t, x: hl
    h_2 = lambda t, x: (4/(9*g))*(np.sqrt(g*hl)-(x - x0)/(2*t))**2
    h_3 = lambda t, x: cm**2/g
    h_4 = lambda t, x: hr

    u_1 = lambda t, x: 0
    u_2 = lambda t, x: (2/3)*((x-x0)/t + np.sqrt(g*hl))
    u_3 = lambda t, x: 2*(np.sqrt(g*hl)-cm)
    u_4 = lambda t, x: 0

    h = np.where(
        x <= xat(t), h_1(t,x), np.where(
            (xat(t) <= x) & (x <= xbt(t)), h_2(t,x), np.where(
                (xbt(t) <= x) & (x <= xct(t)), h_3(t,x), np.where(
                    xct(t) <= x, h_4(t,x), h_4(t,x)
                )
            )
        )
    )

    u = np.where(
        x <= xat(t), u_1(t,x), np.where(
            (xat(t) <= x) & (x <= xbt(t)), u_2(t,x), np.where(
                (xbt(t) <= x) & (x <= xct(t)), u_3(t,x), np.where(
                    xct(t) <= x, u_4(t,x), u_4(t,x)
                )
            )
        )
    )

    return h, u

def dambreak_on_wet_no_friction_analytical_builder(Ll=0, Lr=10, hl=0.005, hr=0.001, x0=5, T=6, dh=0.01):
    
    A = 0.0
    
    x_range = [Ll, Lr]
    y_range = [0,.1]

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = (X <= x0)

    h = np.where(mask, hl, hr)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = 2*np.ones_like(h)

    A_g = A*np.ones_like(X)

    n = np.zeros_like(h)

    x_range = [0, 10]

    y_range = [0,.1]

    inlet_polygon = [[x_range[0]-dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[1]+dh/2],
                     [x_range[0]-dh/2, y_range[1]+dh/2]]
    

    outlet_polygon = [[x_range[1]-dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[1]+dh/2],
                      [x_range[1]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "endTime" : T,
        "cfl" : 0.5, 
        "outFreq" : 1,
        "dh" : dh,
        "h_init": h,
        "u_init": u,
        "v_init": v,
        "z_init": z,
        "roughness": n,
        "A_g" : A_g,
        "boundaries": {
            "inlet": {
                "type": "transmissive_bounds",
                "polygon": inlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.0,1.0,0.0,0.0],
                "normal": [-1.0,0.0]
            },
            "outlet": {
                "type": "transmissive_bounds",
                "polygon": outlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.0, 0.0, 0.0, 0.0],
                "normal": [1.0,0.0]
            }
        },
        "X": X,
        "Y": Y
    }

    return params

def ideal_case_ACM_FCM_paper_analytical(x, t=10, dh=0.1,):
    A = 0.005           
    alpha = 0.005
    beta = 0.005
    gamma = 1
    q0 = 3

    #x_range = [0, 7]

    #x = np.arange(x_range[0], x_range[1]+dh, dh)
    u_func = lambda x: np.sqrt(((alpha*x + beta)/(A))**(2/3))
    q = q0*np.ones_like(x)
    u = u_func(x)
    h = q/u
    z0 = -(u**3 + 2*g*q)/(2*g*u) + gamma

    zt = z0 - alpha*t

    return zt, h, x, q

def ideal_case_ACM_FCM_paper_builder(T=10, dh=0.1):

    # ideal case parameters https://doi.org/10.1016/j.advwatres.2021.103931
    A = 0.005           
    alpha = 0.005
    beta = 0.005
    gamma = 1
    q0 = 3
    
    x_range = [0, 7]
    y_range = [0, dh]

    inlet_polygon = [[x_range[0]-dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[1]+dh/2],
                     [x_range[0]-dh/2, y_range[1]+dh/2]]
    

    outlet_polygon = [[x_range[1]-dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[1]+dh/2],
                      [x_range[1]-dh/2, y_range[1]+dh/2]]

    u_func = lambda x: ((alpha*x + beta)/A)**(1/3)

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")
    q = q0*np.ones_like(X)
    u = u_func(X)
    v = np.zeros_like(u)
    h = q/u
    g = 9.81
    z = -(u**3 + 2*g*q)/(2*g*u) + gamma

    zt_inflow = lambda t: gamma - (u_func(x_range[0])**3 + 2*g*q0)/(2*g*u_func(x_range[0])) - alpha*t
    zt_outflow = lambda t: gamma - (u_func(x_range[1])**3 + 2*g*q0)/(2*g*u_func(x_range[1])) - alpha*t

    A_g = A*np.ones_like(X)

    n = np.zeros_like(h)

    h_outflow = q0/u_func(x_range[1])
    h_inflow = q0/u_func(x_range[0])

    params = {
        "endTime" : T,
        "outFreq" : T,
        "cfl" : 1, 
        "dh" : dh,
        "h_init": h,
        "u_init": u,
        "v_init": np.zeros_like(v),
        "z_init": z,
        "roughness": n,
        "A_g" : A_g,
        "boundaries": {
            "inlet": {
                "type": "Berthon_bounds",
                "polygon": inlet_polygon,
                #         [h, q, z, qb] 
                "values": [h_inflow, q0, zt_inflow, 0],
                "normal": [1.0, 0.0]
            },
            "outlet": {
                "type": "normal_flow_depth",
                "polygon": outlet_polygon,
                #         [h, q, z, qb] 
                "values": [h_outflow, q0, zt_outflow, 0],
                "normal": [1.0, 0.0]
            }
        },
        "X": X,
        "Y": Y
    }

    return params

def symmetrical_dambreak_exner(G=0.001, T=1):
    x_range = [-3, 3]
    y_range = [0,0.01]
    dh = 0.01
    xi = 0.4

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = (X <= 0.5) & (X >= -0.5)
    h = np.where(mask, 1.0, 0.2)
        
    n = np.zeros_like(h)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = np.ones_like(h)
    A_g = (1/(1-xi))*G*np.ones_like(h)

    inlet_polygon = [[x_range[0]-dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[1]+dh/2],
                     [x_range[0]-dh/2, y_range[1]+dh/2]]  

    outlet_polygon = [[x_range[1]-dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[1]+dh/2],
                      [x_range[1]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "endTime" : T,
        "outFreq" : T,
        "cfl" : 1, 
        "dh" : dh,
        "h_init": h,
        "u_init": np.zeros_like(h),
        "v_init": np.zeros_like(h),
        "z_init": z,
        "roughness": n,
        "A_g" : A_g,
        "boundaries": {
            "inlet": {
                "type": "transmissive_bounds",
                "polygon": inlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [-1.0,0.0]
            },
            "outlet": {
                "type": "transmissive_bounds",
                "polygon": outlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.,0.0,0.0,0.0],
                "normal": [1.0,0.0]
            }
        },
        "X": X,
        "Y": Y
    }

    return params

def symmetrical_dambreak_exner_2D():
    x_range = [-6, 6]
    y_range = [-6, 6]
    dh = 0.1
    xi = 0.4
    G = 0.01

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    mask = X**2 + Y**2 <= 1.2**2 
    h = np.where(mask, 1.0, 0.2)
        
    n = np.zeros_like(h)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = np.ones_like(h)
    A_g = (1/(1-xi))*G*np.ones_like(h)
    
    polygon1 = [[x_range[0]-dh/2, y_range[0]-dh/2],
                [x_range[0]+dh/2, y_range[0]-dh/2],
                [x_range[0]+dh/2, y_range[1]+dh/2],
                [x_range[0]-dh/2, y_range[1]+dh/2]]

    polygon2 = [[x_range[1]-dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[1]+dh/2],
                [x_range[1]-dh/2, y_range[1]+dh/2]]

    polygon3 = [[x_range[0]-dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]+dh/2],
                [x_range[0]-dh/2, y_range[0]+dh/2]]

    polygon4 = [[x_range[0]-dh/2, y_range[1]-dh/2],
                [x_range[1]+dh/2, y_range[1]-dh/2],
                [x_range[1]+dh/2, y_range[1]+dh/2],
                [x_range[0]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "endTime" : 4,
        "outFreq" : 0.1,
        "cfl" : 0.5, 
        "dh" : dh,
        "h_init": h,
        "u_init": np.zeros_like(h),
        "v_init": np.zeros_like(h),
        "z_init": z,
        "roughness": n,
        "A_g" : A_g,
        "boundaries": {
            "outlet1": {
                "type": "reflective_bounds",
                "polygon": polygon1,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [-1.0,0.0]
            },
            "outlet2": {
                "type": "reflective_bounds",
                "polygon": polygon2,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [1.0,0.0]
            },
            "outlet3": {
                "type": "reflective_bounds",
                "polygon": polygon3,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [0.0,-1.0]
            },
            "outlet4": {
                "type": "reflective_bounds",
                "polygon": polygon4,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [0.0,1.0]
            },
        },
        "X": X,
        "Y": Y
    }

    return params

def partial_dam_breach():
    x_range = [0, 200]
    y_range = [0, 200]
    dh = 1
    xi = 0.4
    G = 0.01

    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)
    X, Y = np.meshgrid(x, y, indexing="xy")

    mask_z = ((X <= 102) & (X >= 98)) & ((Y <= 85) | (Y >= 115))

    mask = (X < 100)
    h = np.where(mask, 4.0, 1.0)
        
    n = np.zeros_like(h)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    z = np.ones_like(h)

    z[mask_z] = 100
    h[mask_z] = 0
    
    A_g = (1/(1-xi))*G*np.ones_like(h)
    
    polygon1 = [[x_range[0]-dh/2, y_range[0]-dh/2],
                [x_range[0]+dh/2, y_range[0]-dh/2],
                [x_range[0]+dh/2, y_range[1]+dh/2],
                [x_range[0]-dh/2, y_range[1]+dh/2]]

    polygon2 = [[x_range[1]-dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[1]+dh/2],
                [x_range[1]-dh/2, y_range[1]+dh/2]]

    polygon3 = [[x_range[0]-dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]-dh/2],
                [x_range[1]+dh/2, y_range[0]+dh/2],
                [x_range[0]-dh/2, y_range[0]+dh/2]]

    polygon4 = [[x_range[0]-dh/2, y_range[1]-dh/2],
                [x_range[1]+dh/2, y_range[1]-dh/2],
                [x_range[1]+dh/2, y_range[1]+dh/2],
                [x_range[0]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "endTime" : 4,
        "outFreq" : 0.1,
        "cfl" : 0.5, 
        "dh" : dh,
        "h_init": h,
        "u_init": np.zeros_like(h),
        "v_init": np.zeros_like(h),
        "z_init": z,
        "roughness": n,
        "A_g" : A_g,
        "boundaries": {
            "outlet1": {
                "type": "reflective_bounds",
                "polygon": polygon1,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [-1.0,0.0]
            },
            "outlet2": {
                "type": "transmissive_bounds",
                "polygon": polygon2,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [1.0,0.0]
            },
            "outlet3": {
                "type": "reflective_bounds",
                "polygon": polygon3,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [0.0,-1.0]
            },
            "outlet4": {
                "type": "reflective_bounds",
                "polygon": polygon4,
                #         [h, q, z, qb] 
                "values": [0.0,0.0,0.0,0.0],
                "normal": [0.0,1.0]
            },
        },
        "X": X,
        "Y": Y
    }

    return params

if __name__ == "__main__":

    print("--- Numerical Solution Example ---")
    # Example numerical values (you can change these)
    g = 9.81  # Acceleration due to gravity
    hl = 5.0   # Height
    hr = 0.5   # Radius

    numerical_solutions_cm = solve_polynomial_numerically(hl, hr)

    if numerical_solutions_cm is not None:
        print(f"\nNumerical Solutions for c_m:\n{numerical_solutions_cm}")
        # You might want to filter for real solutions depending on the context
        real_solutions = [sol.real for sol in numerical_solutions_cm if np.isclose(sol.imag, 0) and sol.real > 0]
        print(f"Real positive Solutions for c_m (if any):\n{real_solutions}")
    
    print("Generating analytical solution")

    xmin = 0 
    xmax = 10
    tmin = 0
    tmax = 6

    x = np.linspace(xmin, xmax, 100)
    t = np.linspace(tmin, tmax, 100)

    h, u = dambreak_on_wet_no_friction_analytical(6, x)

    plt.plot(x, h)
    plt.savefig("h_analytical_swashes.png")
    plt.close()
    plt.plot(x, h*u)
    plt.savefig("hu_analytical_swashes.png")    
    plt.close()

    
    