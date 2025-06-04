import numpy as np
import matplotlib.pyplot as plt
from roesolver import compute_dt_2D, roe_solve_2D, compute_dt_SWE, exner_solve_2D

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
            masks[i, j] = point_in_polygon(point, polygon)
            
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
        self.n = self.params["roughness"]

        self.qb_x = self.G*((self.hu/self.h)**2 + (self.hv/self.h)**2)*(self.hu/self.h)
        self.qb_y = self.G*((self.hu/self.h)**2 + (self.hv/self.h)**2)*(self.hv/self.h)

        # define spatial parameters
        self.dt = 1
        self.dx = params["dh"]  # cell width
        self.cellArea = self.dx**2 

        # update fields 
        self.contributions = np.zeros((self.h.shape[0], self.h.shape[1], 3))
        self.contributions_z = np.zeros_like(self.z)
    
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
            self.polygons.append(np.array(self.boundaries[key]["polygon"]))
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

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(self.hu)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.qb_x[self.qb_x.shape[0]//2,:])
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[1].grid(alpha=0.25)
        #plt.colorbar()
        fig.savefig(folder+"/debug_qb_x{}.png".format(i))
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(self.hu)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.qb_y[self.qb_y.shape[0]//2,:])
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[1].grid(alpha=0.25)
        #plt.colorbar()
        fig.savefig(folder+"/debug_qb_y{}.png".format(i))
        plt.close()


        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(self.hu)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.hu[self.hu.shape[0]//2,:])
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[1].grid(alpha=0.25)
        #plt.colorbar()
        fig.savefig(folder+"/debug_hu{}.png".format(i))
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(self.hv)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.hv[self.hv.shape[0]//2,:])
        ax[1].grid(alpha=0.25)
        fig.suptitle("Time: {:.3f}".format(self.etime))
        #plt.colorbar()
        fig.savefig(folder+"/debug_hv{}.png".format(i))
        plt.close()

        
        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(self.h)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.h[self.h.shape[0]//2,:]+self.z[self.z.shape[0]//2,:],label="h+z")
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.z[self.z.shape[0]//2,:],label="z")
        ax[1].grid(alpha=0.25)
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[1].legend()
        #plt.colorbar()
        fig.savefig(folder+"/debug_h+z{}.png".format(i))
        plt.close()


        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(self.z)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.z[self.z.shape[0]//2,:])
        ax[1].grid(alpha=0.25)
        fig.suptitle("Time: {:.3f}".format(self.etime))
        #plt.colorbar()
        fig.savefig(folder+"/debug_hz{}.png".format(i))
        plt.close()


    def _apply_boundaries(self):
        # This uses the Python 3.10+ match-case statement.
        for i, key in enumerate(self.boundaries.keys()):
            match self.boundaries[key]["type"]:
                case "constant_flux":
                    qx = self.boundary_values[i][1] * self.normals[i][0]
                    qy = self.boundary_values[i][1] * self.normals[i][1]
                    
                    # NumPy: direct boolean indexing for updates
                    self.hu[self.masks[i]] = qx
                    self.hv[self.masks[i]] = qy
                    
                    u = qx/self.h[self.masks[i]]
                    v = qy/self.h[self.masks[i]]

                    self.qb_x[self.masks[i]] = self.G[self.masks[i]]*(u**2+v**2)*u
                    self.qb_y[self.masks[i]] = self.G[self.masks[i]]*(u**2+v**2)*v

                case "normal_flow_depth":
                    self.h[self.masks[i]] = self.boundary_values[i][0]
                    qx = self.boundary_values[i][1] * self.normals[i][0]
                    qy = self.boundary_values[i][1] * self.normals[i][1]
                    
                    self.hu[self.masks[i]] = qx
                    self.hv[self.masks[i]] = qy

                    u = qx/self.h[self.masks[i]]
                    v = qy/self.h[self.masks[i]]

                    self.qb_x[self.masks[i]] = self.G[self.masks[i]]*(u**2+v**2)*u
                    self.qb_y[self.masks[i]] = self.G[self.masks[i]]*(u**2+v**2)*v


                case "transmissive_bedload":
                    normal = self.normals[i]
                    # NumPy equivalent of np.rint and np.array
                    shift = -np.rint(np.flip(np.array(normal))).astype(int) # shift inward
                    
                    # NumPy equivalent of np.argwhere
                    boundary_cells = np.argwhere(self.masks[i])
                    interior_cells = boundary_cells + shift
                    
                    Ny, Nx = self.masks.shape[1:]
                    
                    # NumPy equivalent of np.clip
                    interior_cells = np.clip(interior_cells, a_min=np.array([0, 0]), a_max=np.array([Ny - 1, Nx - 1]))
                    
                    boundary_x, boundary_y = boundary_cells[:, 0], boundary_cells[:, 1]
                    interior_x, interior_y = interior_cells[:, 0], interior_cells[:, 1]

                    self.boundary_indices = (boundary_x, boundary_y)
                    self.boundary_interior_map = (interior_x, interior_y)

                    # NumPy: direct advanced indexing for updates
                    self.qb_x[self.boundary_indices] = self.qb_x[self.boundary_interior_map]
                    self.qb_y[self.boundary_indices] = self.qb_y[self.boundary_interior_map]

                    self.z[self.boundary_indices] = self.z[self.boundary_interior_map]

                case "transmissive_flux":
                    normal = self.normals[i]
                    shift = -np.rint(np.flip(np.array(normal))).astype(int) # shift inward
                    
                    boundary_cells = np.argwhere(self.masks[i])
                    interior_cells = boundary_cells + shift
                    
                    Ny, Nx = self.masks.shape[1:]
                    interior_cells = np.clip(interior_cells, a_min=np.array([0, 0]), a_max=np.array([Nx - 1, Ny - 1]))
                    
                    boundary_x, boundary_y = boundary_cells[:, 0], boundary_cells[:, 1]
                    interior_x, interior_y = interior_cells[:, 0], interior_cells[:, 1]

                    self.boundary_indices = (boundary_x, boundary_y)
                    self.boundary_interior_map = (interior_x, interior_y)

                    self.h[self.boundary_indices] = self.h[self.boundary_interior_map]
                    self.hu[self.boundary_indices] = self.hu[self.boundary_interior_map]
                    self.hv[self.boundary_indices] = self.hv[self.boundary_interior_map]
                
                case "transmissive_bounds":
                    normal = self.normals[i]
                    # NumPy equivalent of np.flip
                    shift = -np.rint(np.flip(np.array(normal))).astype(int) # shift inward
                    
                    boundary_cells = np.argwhere(self.masks[i])
                    interior_cells = boundary_cells + shift
                    
                    # Consistent with your original JAX code's indexing for this case
                    Ny, Nx = self.masks.shape[1:] 

                    interior_cells = np.clip(interior_cells, a_min=np.array([0, 0]), a_max=np.array([Ny - 1, Nx - 1]))
                    
                    # NumPy equivalent of creating a boolean mask from indices
                    interior_mask = np.zeros_like(self.masks[i], dtype=bool)
                    interior_mask[interior_cells[:,0], interior_cells[:,1]] = True

                    boundary_y, boundary_x = boundary_cells[:, 0], boundary_cells[:, 1]
                    interior_y, interior_x = interior_cells[:, 0], interior_cells[:, 1]

                    self.boundary_indices = (boundary_y, boundary_x)
                    self.boundary_interior_map = (interior_y, interior_x)

                    # NumPy: direct boolean indexing for updates
                    self.h[self.masks[i]] = self.h[interior_mask]
                    self.hu[self.masks[i]] = self.hu[interior_mask]
                    self.hv[self.masks[i]] = self.hv[interior_mask]
                    self.qb_x[self.masks[i]] = self.qb_x[interior_mask]
                    self.qb_y[self.masks[i]] = self.qb_y[interior_mask]
                    self.z[self.masks[i]] = self.z[interior_mask]
    
    def step_hydro(self):
        self.contributions= roe_solve_2D(
           self.contributions, self.h, self.hu, self.hv, self.z, self.G, self.n, self.dx
        )

        self.dt *= self.cfl

        dh =  self.contributions[..., 0]
        dhu = self.contributions[..., 1]
        dhv = self.contributions[..., 2]

        self.h  -= self.dt*dh/self.dx
        self.hu -= self.dt*dhu/self.dx
        self.hv -= self.dt*dhv/self.dx

        self.contributions *= 0
    
    def step_exner(self):
        self.contributions_z, self.qb_x, self.qb_y = exner_solve_2D(
            self.contributions_z, self.h, self.hu, self.hv, self.z, self.qb_x, self.qb_y, self.G, self.n, self.dx
        )

        self.z -= self.dt*self.contributions_z/self.dx
        self.contributions_z *= 0

    def compute_dt(self):
        self.dt = compute_dt_2D(self.h, self.hu, self.hv, self.z, self.G, self.dx)
        #self.dt = compute_dt_SWE(self.h, self.hu, self.hv, self.dx)
        self.dt *= self.cfl

    def evolve(self):
        iters = 0 
        while self.etime < self.endTime:
            
            self.compute_dt()
            self.step_hydro()
            self._apply_boundaries()
            #self.step_exner()

            if iters % 10 == 0: 
                print("Iteration: {} - Time: {:.9f}".format(iters, self.etime))
                print("Timestep: {:.6f}".format(self.dt))
                self._debug_boundaries(iters)

            if self.etime+self.dt > self.endTime:
                self.dt = self.endTime - self.etime

            self.etime += self.dt

            iters += 1


if __name__ == "__main__":

    # ideal case parameters https://doi.org/10.1016/j.advwatres.2021.103931
    A = 0.01           
    alpha = 0.005
    beta = 0.005
    gamma = 2
    
    x_range = [0, 10]
    y_range = [0,.1]

    u_func = lambda x: np.sqrt(((alpha*x + beta)/(A))**(2/3))

    dh = 0.1
    x = np.arange(x_range[0], x_range[1]+dh, dh)
    y = np.arange(y_range[0], y_range[1]+dh, dh)

    X, Y = np.meshgrid(x, y, indexing="xy")
    q = np.ones_like(X)
    u = u_func(X)
    v = np.zeros_like(u)
    h = q/u
    g = 9.81
    z = -(u**3 + 2*g*q)/(2*g*u) + gamma

    A_g = A*np.ones_like(X)

    n = np.zeros_like(h)

    fig, ax = plt.subplots(2,1,figsize=(12,5))
    ax[0].imshow(h)
    ax[1].plot(X[X.shape[0]//2,:], h[h.shape[0]//2,:]+z[z.shape[0]//2,:])
    #plt.colorbar()
    fig.savefig("h_init.png")
    plt.close()

    fig, ax = plt.subplots(2,1,figsize=(12,5))
    ax[0].imshow(z)
    ax[1].plot(X[X.shape[0]//2,:], z[z.shape[0]//2,:])
    #plt.colorbar()
    fig.savefig("z_init.png")
    plt.close()
    
    """
    #    dambreak
    x_range = [-6, 6]
    y_range = [0,0.02]
    dh = 0.01
    xi = 0.4
    G = 0.001

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
    """
    

    inlet_polygon = [[x_range[0]-dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[0]-dh/2],
                     [x_range[0]+dh/2, y_range[1]+dh/2],
                     [x_range[0]-dh/2, y_range[1]+dh/2]]
    

    outlet_polygon = [[x_range[1]-dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[0]-dh/2],
                      [x_range[1]+dh/2, y_range[1]+dh/2],
                      [x_range[1]-dh/2, y_range[1]+dh/2]]
    
    params = {
        "endTime" : 10,
        "cfl" : 0.5, 
        "dh" : dh,
        "h_init": h,
        "u_init": u,
        "v_init": np.zeros_like(v),
        "z_init": z,
        "roughness": n,
        "A_g" : A_g,
        "boundaries": {
            "inlet": {
                "type": "constant_flux",
                "polygon": inlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.0,1.0,0.0,0.0],
                "normal": [1.0,0.0]
            },
            "outlet": {
                "type": "normal_flow_depth",
                "polygon": outlet_polygon,
                #         [h, q, z, qb] 
                "values": [0.5665,1.0,0.0,0.0],
                "normal": [1.0,0.0]
            },
            "outlet_hydro": {
                "type": "transmissive_bedload",
                "polygon": outlet_polygon,
                "values": [0.0,0.0,0.0,0.0],
                "normal": [1.0,0.0]
            },
        },
        "X": X,
        "Y": Y
    }

    exnersim = SWEExnerSim(params)
    exnersim.evolve()

    fig, ax = plt.subplots(2,1,figsize=(12,5))
    ax[0].imshow(h)
    ax[1].plot(X[X.shape[0]//2,:], exnersim.h[h.shape[0]//2,:]+exnersim.z[z.shape[0]//2,:])
    #plt.colorbar()
    fig.savefig("h_evolved.png")
    plt.close()

    fig, ax = plt.subplots(2,1,figsize=(12,5))
    ax[0].imshow(z)
    ax[1].plot(X[X.shape[0]//2,:], exnersim.z[exnersim.z.shape[0]//2,:])
    #plt.colorbar()
    fig.savefig("z_evolved.png")
    plt.close()

    