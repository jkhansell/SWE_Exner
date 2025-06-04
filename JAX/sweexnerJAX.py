import numpy as np
import matplotlib.pyplot as plt


g = 9.81

def point_in_polygon(point, polygon):
    x, y = point
    xi, yi = polygon[:, 0], polygon[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)

    # Ray casting test
    intersect = ((yi > y) != (yj > y)) & \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
    
    return np.sum(intersect) % 2 == 1  # odd = inside


def compute_label_mask(points, polygons):
    num_points = points.shape[0]
    num_polygons = len(polygons)
    
    masks = np.zeros((num_polygons, num_points), dtype=bool)

    # Equivalent to jax.vmap(test_poly)(polygons)
    # We iterate over each polygon
    for i, polygon in enumerate(polygons):
        # Equivalent to jax.vmap(lambda pt: point_in_polygon(pt, polygon))(points)
        # We iterate over each point for the current polygon
        for j, point in enumerate(points):
            masks[i, j] = point_in_polygon_numpy(point, polygon)
            
    return masks

class SWEExnerSim:
    def __init__(self, params):

        self.params = params

        self._initialize_simparams()
        self._initialize_fields()
        self._initialize_boundaries()
        self._apply_boundaries()

    def _initialize_simparams(self):
        self.endTime = self.params["endTime"]
        self.cfl = self.params["cfl"]
        self.etime = 0.0

    def _initialize_fields(self):
        # input fields
        
        self.h = self.params["h_init"]
        self.hu = self.h*self.params["u_init"]
        self.hv = self.h*self.params["v_init"]

        self.z = self.params["z_init"]
        self.G = self.params["A_g"]

        self.qb_x = self.G*((self.hu/self.h)**2 + (self.hv/self.h)**2)*(self.hu/self.h)
        self.qb_y = self.G*((self.hu/self.h)**2 + (self.hv/self.h)**2)*(self.hv/self.h)

        # define spatial parameters
        self.dx = params["dh"]  # cell width
        self.cellArea = self.dx**2 

        # update fields 
        self.contributions = jnp.zeros((self.h.shape[0], self.h.shape[1], 3))
        self.contributions_z = jnp.zeros_like(self.z)

    def _initialize_boundaries(self):
        self.X = self.params["X"]
        self.Y = self.params["Y"]
        self.boundaries = self.params["boundaries"]
        points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        self.polygons = []
        self.boundary_values = []
        self.normals = []

        for i, key in enumerate(self.boundaries.keys()):
            #get points inside
            self.polygons.append(jnp.array(self.boundaries[key]["polygon"]))
            self.boundary_values.append(self.boundaries[key]["values"])
            self.normals.append(self.boundaries[key]["normal"])
            
        # --- Pad polygons to same length ---
        max_vertices = max(p.shape[0] for p in self.polygons)
        polygons_padded = np.stack([
            np.pad(p, ((0, max_vertices - p.shape[0]), (0, 0))) for p in self.polygons
        ])

        self.masks = compute_label_mask(points, polygons_padded).reshape(len(self.polygons), self.X.shape[0], self.X.shape[1])
    
    def _debug_boundaries(self, i):
        folder = "debug"

        plt.figure(figsize=(12,5))
        plt.plot(self.X[5,:], self.hu[5,:])
        plt.title("Time: {:.3f}".format(self.etime))
        #plt.colorbar()
        plt.savefig(folder+"/debug_hu{}.png".format(i))
        plt.close()
        
        plt.figure(figsize=(12,5))
        plt.title("Time: {:.3f}".format(self.etime))
        plt.plot(self.X[5,:],self.hv[5,:])
        #plt.colorbar()
        plt.savefig(folder+"/debug_hv{}.png".format(i))
        plt.close()

        plt.figure(figsize=(12,5))
        plt.title("Time: {:.3f}".format(self.etime))
        plt.plot(self.X[5,:],self.qb_x[5,:])
        #plt.colorbar()
        plt.savefig(folder+"/debug_qb_x{}.png".format(i))
        plt.close()

        plt.figure(figsize=(12,5))
        plt.title("Time: {:.3f}".format(self.etime))
        plt.plot(self.X[5,:],self.qb_y[5,:])
        #plt.colorbar()
        plt.savefig(folder+"/debug_qb_y{}.png".format(i))
        plt.close()

        plt.figure(figsize=(12,5))
        plt.title("Time: {:.3f}".format(self.etime))
        plt.plot(self.X[5,:],self.h[5,:])
        #plt.colorbar()
        plt.savefig(folder+"/debug_h{}.png".format(i))
        plt.close()

        plt.figure(figsize=(12,5))
        plt.title("Time: {:.3f}".format(self.etime))
        plt.plot(self.X[5,:],self.z[5,:])
        #plt.colorbar()
        plt.savefig(folder+"/debug_z{}.png".format(i))
        plt.close()

            
    def _apply_boundaries(self):
        for i, key in enumerate(self.boundaries.keys()):
            match self.boundaries[key]["type"]:
                case "constant_flux":
                    qx = self.boundary_values[i][1]*self.normals[i][0]
                    qy = self.boundary_values[i][1]*self.normals[i][1]
                    
                    self.hu = jnp.where(self.masks[i], qx, self.hu)
                    self.hv = jnp.where(self.masks[i], qy, self.hv)
                    
                    qb_x = self.boundary_values[i][3]*self.normals[i][0]
                    qb_y = self.boundary_values[i][3]*self.normals[i][1]
                    
                    self.qb_x = jnp.where(self.masks[i], qb_x, self.qb_x) 
                    self.qb_y = jnp.where(self.masks[i], qb_y, self.qb_y) 

                case "normal_flow_depth":
                    self.h = jnp.where(self.masks[i], self.boundary_values[i][0], self.h)
                    qx = self.boundary_values[i][1]*self.normals[i][0]
                    qy = self.boundary_values[i][1]*self.normals[i][1]
                    
                    self.hu = jnp.where(self.masks[i], qx, self.hu)
                    self.hv = jnp.where(self.masks[i], qy, self.hv)

                case "transmissive_bedload":
                    normal = self.normals[i]  # these might be floats like (0.707, 0.707)
                    # Turn into integer shift by rounding
                    shift = -jnp.rint(jnp.array(normal)).astype(int)  # shift inward
                    # Now shift to get interior cells
                    boundary_cells = jnp.argwhere(self.masks[i])
                    interior_cells = boundary_cells + shift  # shape (n_boundary_cells, 2)
                    Nx, Ny = self.masks.shape[1:]
                    interior_cells = jnp.clip(interior_cells, a_min=jnp.array([0, 0]), a_max=jnp.array([Nx - 1, Ny - 1]))
                    boundary_x, boundary_y = boundary_cells[:, 0], boundary_cells[:, 1]
                    interior_x, interior_y = interior_cells[:, 0], interior_cells[:, 1]

                    self.boundary_indices = (boundary_x, boundary_y)
                    self.boundary_interior_map = (interior_x, interior_y)

                    self.qb_x = self.qb_x.at[self.boundary_indices].set(self.qb_x[self.boundary_interior_map])
                    self.qb_y = self.qb_y.at[self.boundary_indices].set(self.qb_y[self.boundary_interior_map])

                case "transmissive_flux":
                    normal = self.normals[i]  # these might be floats like (0.707, 0.707)
                    # Turn into integer shift by rounding
                    shift = -jnp.rint(jnp.array(normal)).astype(int)  # shift inward
                    # Now shift to get interior cells
                    boundary_cells = jnp.argwhere(self.masks[i])
                    interior_cells = boundary_cells + shift  # shape (n_boundary_cells, 2)
                    Nx, Ny = self.masks.shape[1:]
                    interior_cells = jnp.clip(interior_cells, a_min=jnp.array([0, 0]), a_max=jnp.array([Nx - 1, Ny - 1]))
                    boundary_x, boundary_y = boundary_cells[:, 0], boundary_cells[:, 1]
                    interior_x, interior_y = interior_cells[:, 0], interior_cells[:, 1]

                    self.boundary_indices = (boundary_x, boundary_y)
                    self.boundary_interior_map = (interior_x, interior_y)

                    self.h = self.h.at[self.boundary_indices].set(self.h[self.boundary_interior_map])
                    self.hu = self.hu.at[self.boundary_indices].set(self.hu[self.boundary_interior_map])
                    self.hv = self.hv.at[self.boundary_indices].set(self.hv[self.boundary_interior_map])
                
                case "transmissive_bounds":
                    normal = self.normals[i]  # these might be floats like (0.707, 0.707)
                    # Turn into integer shift by rounding
                    shift = -jnp.rint(jnp.flip(jnp.array(normal))).astype(int)  # shift inward
                    # Now shift to get interior cells
                    boundary_cells = jnp.argwhere(self.masks[i])
                    interior_cells = boundary_cells + shift  # shape (n_boundary_cells, 2)

                    Ny, Nx = self.masks.shape[1:]

                    interior_cells = jnp.clip(interior_cells, a_min=jnp.array([0, 0]), a_max=jnp.array([Ny - 1, Nx - 1]))
                    interior_mask = jnp.zeros_like(self.masks[i])
                    interior_mask = interior_mask.at[(interior_cells[:,0], interior_cells[:,1])].set(True)


                    boundary_y, boundary_x = boundary_cells[:, 0], boundary_cells[:, 1]
                    interior_y, interior_x = interior_cells[:, 0], interior_cells[:, 1]

                    self.boundary_indices = (boundary_x, boundary_y)
                    self.boundary_interior_map = (interior_x, interior_y)

                    self.h = self.h.at[self.masks[i]].set(self.h[interior_mask])
                    self.hu = self.hu.at[self.masks[i]].set(self.hu[interior_mask])
                    self.hv = self.hv.at[self.masks[i]].set(self.hv[interior_mask])
                    self.qb_x = self.qb_x.at[self.masks[i]].set(self.qb_x[interior_mask])
                    self.qb_y = self.qb_y.at[self.masks[i]].set(self.qb_y[interior_mask])
                    self.z = self.z.at[self.masks[i]].set(self.z[interior_mask])

    def step_hydro(self):
        self.contributions = roe_solve_2D_jit(
           self.contributions, self.h, self.hu, self.hv, self.z, self.G, self.dx, self.dt
        )
        

        dh =  self.contributions[..., 0]
        dhu = self.contributions[..., 1]
        dhv = self.contributions[..., 2]

        self.h  = self.h.at[:].add(-self.dt*dh/self.dx) 
        self.hu = self.hu.at[:].add(-self.dt*dhu/self.dx)
        self.hv = self.hv.at[:].add(-self.dt*dhv/self.dx)

        self.contributions = self.contributions.at[:].set(0.0)

    """def step_exner(self):
        self.contributions_z = exner_solve_2D_jit(
            self.contributions_z, self.h, self.hu, self.hv, self.z, self.G, self.dx
        )

        z_new = self.z - self.dt*self.contributions_z/(self.dx**2)
        self.z = z_new
        self.contributions_z = self.contributions_z.at[:].set(0.0)"""


    def evolve(self):
        iters = 0 
        while self.etime < self.endTime:
            self.dt = compute_dt_SWE(self.h, self.hu, self.hv, self.dx) 
            self.dt *= self.cfl

            self.step_hydro()           # evolve hydrodynamic variables
            self._apply_boundaries()

            if iters % 1 == 0: 
                print("Iteration: {} - Time: {:.9f}".format(iters, self.etime))
                print("Timestep: {:.6f}".format(self.dt))
                self._debug_boundaries(iters)      

            #self._apply_boundaries()
            #self.step_exner()           # bedload transport 

            # assign dt for last time step
            if self.etime+self.dt > self.endTime:
                self.dt = self.endTime - self.etime

            self.etime += self.dt

            iters += 1


if __name__ == "__main__":

    # ideal case parameters https://doi.org/10.1016/j.advwatres.2021.103931
    """A = 0.01           
    alpha = 0.005
    beta = 0.005
    gamma = 2
    
    x_range = [0, 10]
    y_range = [0,2]

    u_func = lambda x: jnp.sqrt(((alpha*x + beta)/(A))**(2/3))

    dh = 0.1
    x = jnp.arange(x_range[0], x_range[1]+dh, dh)
    y = jnp.arange(y_range[0], y_range[1]+dh, dh)

    X, Y = jnp.meshgrid(x, y, indexing="xy")
    q = jnp.ones_like(X)
    u = u_func(X)
    v = jnp.zeros_like(u)
    h = q/u
    g = 9.81
    z = -(u**3 + 2*g*q)/(2*g*u) + gamma

    A_g = A*jnp.ones_like(X)

    plt.figure(figsize=(12,4))
    plt.plot(X[5,:], z[5,:])
    plt.grid(alpha=0.25)
    plt.savefig("z_init.png")
    plt.close()

    plt.figure(figsize=(12,4))
    plt.plot(X[5,:], h[5,:])
    plt.grid(alpha=0.25)
    plt.savefig("h_init.png")
    plt.close()
    
    plt.figure(figsize=(12,4))
    plt.plot(X[5,:], u[5,:])
    plt.grid(alpha=0.25)
    plt.savefig("u_init.png")
    plt.close()

    plt.figure(figsize=(12,4))
    plt.plot(X[5,:], h[5,:] + z[5,:])
    plt.grid(alpha=0.25)
    plt.savefig("h+z_init.png")
    plt.close()"""

    x_range = [-1, 1]
    y_range = [0,1]
    dh = 0.01
    xi = 0.4
    G = 0.01

    x = jnp.arange(x_range[0], x_range[1]+dh, dh)
    y = jnp.arange(y_range[0], y_range[1]+dh, dh)
    X, Y = jnp.meshgrid(x, y, indexing="xy")

    mask = (X <= 0.5) & (X >= -0.5)
    h_break = jnp.where(mask, 1.0 - 1e-9 * X, 0.5) # Add a tiny slope
    h = h_break
        
    u = jnp.zeros_like(h)
    v = jnp.zeros_like(h)
    z = jnp.ones_like(h)
    A_g = (1/(1-xi))*G*jnp.ones_like(h)
    

    inlet_polygon = [[x_range[0]-dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[1]+dh/2],
                     [x_range[0]-dh/2, y_range[1]+dh/2]]
    

    outlet_polygon = [[x_range[1]-dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[1]+dh/2],
                      [x_range[1]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "endTime" : 1,
        "cfl" : 0.5,
        "dh" : dh,
        "h_init": h,
        "u_init": u,
        "v_init": v,
        "z_init": z,
        "A_g" : A_g,
        "boundaries": {
            "inlet": {
                "type": "transmissive_bounds",
                "polygon": inlet_polygon,
                #          [h, q, z, qb] 
                 "values": [0.0,0.0,0.0,0.0],
                 "normal": [-1.0,0.0] 
            },
            "outlet": {
                "type": "transmissive_bounds",
                "polygon": outlet_polygon,
                "values": [0.0,0.0,0.0,0.0],
                "normal": [1.0,0.0]
            },
            #"outlet_hydro": {
            #    "type": "transmissive_flux",
            #    "polygon": outlet_polygon,
            #    "values": [0.0,0.0,0.0,0.0],
            #    "normal": [1.0,0.0]
            #}
        },
        "X": X,
        "Y": Y
    }

    exnersim = SWEExnerSim(params)
    exnersim.evolve()

    """plt.figure(figsize=(12,4))
    plt.imshow(exnersim.z)
    plt.savefig("zfield.png")
    plt.close()

    plt.figure(figsize=(12,4))
    plt.plot(X[5,:], z[5,:],linestyle="dotted", label="Initial")
    plt.plot(X[5,:], (z - alpha*10)[5,:], label="Exact 10s")
    plt.plot(X[5,:], exnersim.z[5,:], label="Sim 10s")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.savefig("z_evolved.png")
    plt.close()
    
    plt.figure(figsize=(12,4))
    plt.plot(X[5,:], h[5,:] + z[5,:], linestyle="dotted", label="Initial")
    plt.plot(X[5,:], (h + (z - alpha*10))[5,:], label="Exact 10s")
    plt.plot(X[5,:], exnersim.h[5,:] + exnersim.z[5,:], label="Sim 10s")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.savefig("h+z_evolved.png")
    plt.close()"""