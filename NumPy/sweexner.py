import numpy as np
import matplotlib.pyplot as plt
from roesolver import compute_dt_2D, roe_solve_2D, compute_dt_SWE, exner_solve_2D
import case_builder

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
        self.outFreq = self.params["outFreq"]          # desired spacing, e.g. 0.10
        self._eps = 1e-12     # tolerance already used elsewhere

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
        self.dx = params["dh"]          # cell width
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

        self.qb_x = self.G*((self.hu/self.h)**2+(self.hv/self.h)**2)*(self.hu/self.h)
        self.qb_y = self.G*((self.hu/self.h)**2+(self.hv/self.h)**2)*(self.hv/self.h)
        folder = "debug"

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        im = ax[0].imshow(self.qb_x)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.qb_x[self.qb_x.shape[0]//2,:])
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[1].grid(alpha=0.25)
        fig.savefig(folder+"/debug_qb_x{}.png".format(i))
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        im = ax[0].imshow(self.qb_y)
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.qb_y[self.qb_y.shape[0]//2,:])
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[1].grid(alpha=0.25)
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
        ax[1].set_xlim(-2.75, 2.75)
        ax[1].set_ylim(1.18, 1.8)
        #plt.colorbar()
        fig.savefig(folder+"/debug_h+z{}.png".format(i))
        plt.close()

    
        fig, ax = plt.subplots(2,1,figsize=(12,6.5),sharex=True)
        ax[0].plot(self.X[self.X.shape[0]//2,:], self.h[self.h.shape[0]//2,:]+self.z[self.z.shape[0]//2,:],c="blue",
                    marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax[0].grid(alpha=0.25)
        ax[0].set_xlim(-2.75, 2.75)
        ax[0].set_ylim(1.18, 1.8)
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[0].set_ylabel("Free Surf level (m)")
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.z[self.z.shape[0]//2,:],c="blue",
                    marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax[1].grid(alpha=0.25)
        ax[1].set_xlim(-2.75, 2.75)
        ax[1].set_ylim(0.9, 1.05)
        ax[1].set_ylabel("Bed level (m)")
        fig.suptitle("Time: {:.3f}".format(self.etime))
        #plt.colorbar()
        fig.savefig(f"{folder}/debug_hz{i:06d}.png")
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

                    self.qb_x[self.masks[i]] = self.boundary_values[i][3]  #self.G[self.masks[i]]*(u**2+v**2)*u
                    self.qb_y[self.masks[i]] = self.boundary_values[i][3] #self.G[self.masks[i]]*(u**2+v**2)*v

                case "normal_flow_depth":
                    h = self.boundary_values[i][0]
                    qx = self.boundary_values[i][1] * self.normals[i][0]
                    qy = self.boundary_values[i][1] * self.normals[i][1]

                    self.h[self.masks[i]] = h
                    self.hu[self.masks[i]] = qx
                    self.hv[self.masks[i]] = qy

                    u = qx/h
                    v = qy/h

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

    def compute_dt(self):
        self.dt = compute_dt_2D(self.h, self.hu, self.hv, self.z, self.G, self.dx)
        #self.dt = compute_dt_SWE(self.h, self.hu, self.hv, self.dx)
        self.dt *= self.cfl

    def update_sediment_fluxes(self):
        self.qb_x = self.G*((self.hu/self.h)**2 + (self.hv/self.h)**2)*(self.hu/self.h)
        self.qb_y = self.G*((self.hu/self.h)**2 + (self.hv/self.h)**2)*(self.hv/self.h)

    def step_hydro(self):
        self.contributions= roe_solve_2D(
           self.contributions, self.h, self.hu, self.hv, self.z, self.n, self.dx
        )

        dh =  self.contributions[..., 0]
        dhu = self.contributions[..., 1]
        dhv = self.contributions[..., 2]

        self.h  -= self.dt*dh/self.dx
        self.hu -= self.dt*dhu/self.dx
        self.hv -= self.dt*dhv/self.dx

        self.contributions = np.zeros_like(self.contributions)

    def step_exner(self):
        self.contributions_z = exner_solve_2D(
            self.contributions_z, self.h, self.hu, self.hv, self.z, self.G, self.dx
        )

        self.z -= self.dt*self.contributions_z/self.dx
        self.contributions_z = np.zeros_like(self.contributions_z)

    def evolve(self):
        iters = 0 
        while self.etime < self.endTime - self._eps:
            self.compute_dt()

            next_out_time = (int(self.etime / self.outFreq) + 1) * self.outFreq

            next_target = min(next_out_time, self.endTime)
            if self.etime + self.dt >= next_target - self._eps:
                self.dt = next_target - self.etime

            self.step_hydro()
            self.update_sediment_fluxes()
            self._apply_boundaries()
            self.step_exner()

            self.etime += self.dt
            iters += 1

            # Fire debug if we've just landed on a debug multiple
            if abs(self.etime / self.outFreq - round(self.etime / self.outFreq)) < self._eps or iters == 0:
                print(f"Visualizing - Iteration: {iters}  Time: {self.etime:.9f}")
                self._debug_boundaries(iters)

            if iters % 250 == 0:
                print(f"Iteration: {iters}  Time: {self.etime:.9f}")
                print(f"Timestep:  {self.dt:.6f}")
            # Event handling

if __name__ == "__main__":

    params = case_builder.symmetrical_dambreak_exner()

    exnersim = SWEExnerSim(params)
    exnersim.evolve()

    """fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(x, zt+h, label="Analytical")
    ax.plot(x, exnersim.h[exnersim.h.shape[0]//2]+exnersim.z[exnersim.z.shape[0]//2], label="ACM")
    ax.grid(alpha=0.25)
    fig.legend()
    fig.savefig("h+z_evolved.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(x, zt, label="Analytical")
    ax.plot(x, exnersim.z[exnersim.z.shape[0]//2], label="ACM")
    ax.grid(alpha=0.25)
    fig.legend()
    fig.savefig("z_evolved.png")
    plt.close()"""