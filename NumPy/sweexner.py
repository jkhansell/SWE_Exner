import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
# Enable LaTeX rendering globally
plt.rcParams['text.usetex'] = True
# Own modules
import case_builder

# core methods
from roesolver import compute_dt_2D, roe_solve_2D, compute_dt_SWE, exner_solve_2D, wet_dry_correction

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
        self._debug_boundaries(0)

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

        self.update_sediment_fluxes()

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

        folder = "debug"
        """
        fig = plt.figure(figsize=(12, 6.5))  # Adjust overall figure size
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.7, 1])  # 3 rows: 2 for top, 1 for bottom

        # Top plot takes rows 0 and 1
        ax1 = fig.add_subplot(gs[0])
        # Bottom plot takes row 2
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        #fig, ax = plt.subplots(2,1,figsize=(12,6.5),sharex=True)
        ax1.plot(self.X[self.X.shape[0]//2,:], self.h[self.h.shape[0]//2,:]+self.z[self.z.shape[0]//2,:],c="blue", marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax1.grid(alpha=0.25)
        ax1.set_ylabel(r"Free surface level $h+z_b$ [m]", fontsize=14)
        ax1.set_ylim(1.15, 1.625)
        
        ax2.plot(self.X[self.X.shape[0]//2,:], self.z[self.z.shape[0]//2,:],c="blue", marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax2.grid(alpha=0.25)
        ax2.set_ylabel(r"Bed level $z_b$ [m]", fontsize=14)
        ax2.set_xlabel(r"$x$ [m]", fontsize=14)
        ax2.set_ylim(0.55, 1.4)
        ax2.set_xlim(-2.65, 2.65)
        
        fig.suptitle("Time: {:.3f}".format(self.etime), fontsize=14)
        fig.savefig(f"{folder}/debug_hz{self.etime:.3f}.png",dpi=200)
        
        plt.close(fig)

        fig, ax = plt.subplots(2,1,figsize=(12,6.5),sharex=True)
        ax[0].plot(self.X[self.X.shape[0]//2,:], self.hu[self.hu.shape[0]//2,:],c="blue",
                    marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax[0].grid(alpha=0.25)
        #ax[0].set_ylim(1.18, 1.8)
        fig.suptitle("Time: {:.3f}".format(self.etime))
        ax[0].set_ylabel(r"Momentum $x$ $[\mathrm{m}^2/\mathrm{s}]$")
        ax[1].plot(self.X[self.X.shape[0]//2,:], self.hv[self.hv.shape[0]//2,:],c="blue",
                    marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax[1].grid(alpha=0.25)
        #ax[1].set_ylim(0.9, 1.05)
        ax[0].set_ylabel(r"Momentum $y$ $[\mathrm{m}^2/\mathrm{s}]$")
        fig.suptitle("Time: {:.3f}".format(self.etime))
        fig.savefig(f"{folder}/debug_mom{self.etime:.3f}.png",dpi=200)
        plt.close(fig)

        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

        # Plot water depth h
        im0 = ax[0].imshow(self.h, extent=[self.X[0, 0], self.X[0, -1], self.Y[0, 0], self.Y[-1, 0]], cmap="nipy_spectral")
        cbar0 = fig.colorbar(im0, ax=ax[0])
        cbar0.set_label("Water depth $h$")

        # Plot bed height z
        im1 = ax[1].imshow(self.z, extent=[self.X[0, 0], self.X[0, -1], self.Y[0, 0], self.Y[-1, 0]], cmap="nipy_spectral")
        cbar1 = fig.colorbar(im1, ax=ax[1])
        cbar1.set_label("Bed elevation $z_b$")

        # Titles and layout
        fig.suptitle("Time: {:.3f}".format(self.etime))

        # Save the figure
        fig.savefig(f"{folder}/debug_hz{self.etime:.3f}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

        # Plot water depth h
        im0 = ax[0].imshow(self.hu, extent=[self.X[0, 0], self.X[0, -1], self.Y[0, 0], self.Y[-1, 0]], cmap="jet")
        cbar0 = fig.colorbar(im0, ax=ax[0])
        cbar0.set_label("Momentum $x$ $hu$")

        # Plot bed height z
        im1 = ax[1].imshow(self.hv, extent=[self.X[0, 0], self.X[0, -1], self.Y[0, 0], self.Y[-1, 0]], cmap="jet")
        cbar1 = fig.colorbar(im1, ax=ax[1])
        cbar1.set_label("Momentum $y$ $hv$")

        # Titles and layout
        fig.suptitle("Time: {:.3f}".format(self.etime))

        # Save the figure
        fig.savefig(f"{folder}/debug_mom{self.etime:.3f}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
        
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
                
                case "Berthon_bounds":
                    #h = self.boundary_values[i][0]
                    qx = self.boundary_values[i][1] * self.normals[i][0]
                    qy = self.boundary_values[i][1] * self.normals[i][1]

                    #self.h[self.masks[i]] = h
                    self.hu[self.masks[i]] = qx
                    self.hv[self.masks[i]] = qy

                    self.z[self.masks[i]] = self.boundary_values[i][2](self.etime)

                case "normal_flow_depth":
                    #h = self.boundary_values[i][0]
                    qx = self.boundary_values[i][1] * self.normals[i][0]
                    qy = self.boundary_values[i][1] * self.normals[i][1]

                    #self.h[self.masks[i]] = h
                    self.hu[self.masks[i]] = qx
                    self.hv[self.masks[i]] = qy

                    #self.z[self.masks[i]] = self.boundary_values[i][2](self.etime)

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
                    self.z[self.boundary_indices] = self.z[self.boundary_interior_map]
                    #self.h[self.boundary_indices] = self.h[self.boundary_interior_map]

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

                case "reflective_bounds":
                    normal = self.normals[i]  # e.g., [1, 0] or [0, -1]
                    
                    # Shift opposite to normal to find interior cell
                    shift = -np.rint(np.flip(np.array(normal))).astype(int)  
                    
                    boundary_cells = np.argwhere(self.masks[i])
                    interior_cells = boundary_cells + shift

                    Ny, Nx = self.masks.shape[1:] 
                    interior_cells = np.clip(interior_cells, a_min=np.array([0, 0]), a_max=np.array([Ny - 1, Nx - 1]))

                    interior_mask = np.zeros_like(self.masks[i], dtype=bool)
                    interior_mask[interior_cells[:,0], interior_cells[:,1]] = True

                    boundary_y, boundary_x = boundary_cells[:, 0], boundary_cells[:, 1]
                    interior_y, interior_x = interior_cells[:, 0], interior_cells[:, 1]

                    self.boundary_indices = (boundary_y, boundary_x)
                    self.boundary_interior_map = (interior_y, interior_x)

                    # Scalars: copied
                    self.h[self.masks[i]] = self.h[interior_mask]
                    self.z[self.masks[i]] = self.z[interior_mask]
                    
                    self.qb_x[self.masks[i]] = self.qb_x[interior_mask]
                    self.qb_y[self.masks[i]] = self.qb_y[interior_mask]

                    # Momentum: reflect normal component, preserve tangential
                    nx, ny = normal

                    hu_int = self.hu[interior_mask]
                    hv_int = self.hv[interior_mask]

                    # Reflect components
                    self.hu[self.masks[i]] = hu_int * (1 - 2 * nx * nx)
                    self.hv[self.masks[i]] = hv_int * (1 - 2 * ny * ny)

                    # Sediment flux (if vectorial like momentum): handle similarly
                    qb_x_int = self.qb_x[interior_mask]
                    qb_y_int = self.qb_y[interior_mask]
                    
                    self.qb_x[self.masks[i]] = qb_x_int * (1 - 2 * nx * nx)
                    self.qb_y[self.masks[i]] = qb_y_int * (1 - 2 * ny * ny)

    def compute_dt(self):
        self.dt = compute_dt_2D(self.h, self.hu, self.hv, self.z, self.G, self.dx)
        #self.dt = compute_dt_SWE(self.h, self.hu, self.hv, self.dx)
        self.dt *= self.cfl

    def update_sediment_fluxes(self):
        h = self.h + self._eps
        flux_x = self.G*((self.hu/h)**2 + (self.hv/h)**2)*(self.hu/h)
        flux_y = self.G*((self.hu/h)**2 + (self.hv/h)**2)*(self.hv/h)

        self.qb_x = np.where(self.h < self._eps, 0.0, flux_x)
        self.qb_y = np.where(self.h < self._eps, 0.0, flux_y)


    def wet_dry_correction(self):
        self.hu, self.hv = wet_dry_correction(self.h, self.z, self.hu, self.hv, self.masks, 1e-4)

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
            self.contributions_z, self.h, self.hu, self.hv, 
            self.z, self.G, self.dx
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
            self._apply_boundaries()
            self.wet_dry_correction()
            self.update_sediment_fluxes()
            self.step_exner()

            self.etime += self.dt
            iters += 1

            # Fire debug if we've just landed on a debug multiple
            if abs(self.etime / self.outFreq - round(self.etime / self.outFreq)) < self._eps or iters == 0:
                print(f"Visualizing - Iteration: {iters}  Time: {self.etime:.9f}")
                self._debug_boundaries(iters)

            if iters % 1000 == 0:
                print(f"Iteration: {iters}  Time: {self.etime:.9f}")
                print(f"Timestep:  {self.dt:.6f}")
                # Event handling

if __name__ == "__main__":

    params = case_builder.symmetrical_dambreak_exner_2D()
    exnersim = SWEExnerSim(params)
    exnersim.evolve()

    """
    params = case_builder.dambreak_on_wet_no_friction_analytical_builder(T=1, hl=1, hr=0.2)
    x = params["X"][params["X"].shape[0]//2]
    
    exnersim = SWEExnerSim(params)
    exnersim.evolve()

    h, u = case_builder.dambreak_on_wet_no_friction_analytical(1, x, hl=1, hr=0.2)
    h_init, u_init = case_builder.dambreak_on_wet_no_friction_analytical(0, x, hl=1, hr=0.2)
    
    anchor = (0.9, 0.875)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, h, linestyle="dashed", linewidth=3, c="black", label="Analytical")
    ax.plot(x, h_init, linestyle="dotted", linewidth=2, c="gray", label="Initial")
    ax.plot(x, exnersim.h[exnersim.h.shape[0]//2], c="red", label="Numerical")
    ax.grid(alpha=0.25)
    ax.set_ylabel(r"Water depth $h$ $[\mathrm{m}]$", fontsize=14)
    ax.set_xlabel(r"Channel Length $x$ $[\mathrm{m}]$", fontsize=14)
    fig.legend(bbox_to_anchor=anchor, fontsize=12)
    fig.savefig("h_evolved_swashes.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, h*u, linestyle="dashed", linewidth=3, c="black", label="Analytical")
    ax.plot(x, (params["h_init"]*params["u_init"])[params["h_init"].shape[0]//2], linestyle="dotted", linewidth=2, c="gray", label="Initial")
    ax.plot(x, exnersim.hu[exnersim.h.shape[0]//2], c="red", label="Numerical")
    ax.grid(alpha=0.25)
    ax.set_ylabel(r"Momentum $hu$ $[\mathrm{m}^2/\mathrm{s}]$", fontsize=14)
    ax.set_xlabel(r"Channel Length $x$ $[\mathrm{m}]$", fontsize=14)
    fig.legend(bbox_to_anchor=anchor, fontsize=12)
    fig.savefig("hu_evolved_swashes.png", dpi=200)
    plt.close()

    dhs = [0.1, 0.05, 0.025, 0.0125, 0.00625]

    l1_norm_h = [] 
    l2_norm_h = [] 
    linf_norm_h = [] 

    l1_norm_hu = [] 
    l2_norm_hu = [] 
    linf_norm_hu = [] 

    for dh in dhs:
        print(dhs)
        params = case_builder.dambreak_on_wet_no_friction_analytical_builder(T=1, hl=1, hr=0.2, dh=dh)
        x = params["X"][params["X"].shape[0]//2]

        h, u = case_builder.dambreak_on_wet_no_friction_analytical(1, params["X"], hl=1, hr=0.2)
        
        exnersim = SWEExnerSim(params)
        exnersim.evolve()

        h_error = h - exnersim.h

        l1_norm_h.append(dh*dh*np.sum(np.abs(h_error)))
        linf_norm_h.append(np.max(np.abs(h_error)))

        hu_error = h*u - exnersim.hu

        l1_norm_hu.append(dh*dh*np.sum(np.abs(hu_error)))
        linf_norm_hu.append(np.max(np.abs(hu_error)))
 
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(l1_norm_h,   linestyle="dotted", marker=".",  label=r"$L_1$ norm $h$")
    ax.plot(l1_norm_hu,   linestyle="dotted", marker=".", label=r"$L_1$ norm $hu$")
    ax.grid(alpha=0.25, which="both")
    ax.set_ylabel(r"$L_1$ error [m]", fontsize=14)
    ax.set_xlabel(r"$\Delta x$ [m]", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks([i for i in range(len(dhs))])
    ax.set_xticklabels([f"${dh}$" for dh in dhs], fontsize=12)
    fig.legend(bbox_to_anchor=(0.9,0.875), fontsize=12)
    fig.savefig("L1_norms_swashes.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(linf_norm_h, linestyle="dotted", marker=".",  label=r"$L_\infty$ norm $h$")
    ax.plot(linf_norm_hu, linestyle="dotted", marker=".", label=r"$L_\infty$ norm $hu$")
    ax.grid(alpha=0.25, which="both")
    ax.set_ylabel(r"$L_\infty$ error [m]", fontsize=14)
    ax.set_xlabel(r"$\Delta x$ [m]", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks([i for i in range(len(dhs))])
    ax.set_xticklabels([f"${dh}$" for dh in dhs], fontsize=12)
    fig.legend(bbox_to_anchor=(0.9,0.5), fontsize=12)
    fig.savefig("Linf_norms_swashes.png", dpi=200)
    plt.close()

    params = case_builder.ideal_case_ACM_FCM_paper_builder(T=10, dh=0.05)
    zt_init, h_init, x, q_init = case_builder.ideal_case_ACM_FCM_paper_analytical(params["X"],0)

    anchor = (0.85, 0.85)

    exnersim = SWEExnerSim(params)
    exnersim.evolve()

    zt, h, x, q = case_builder.ideal_case_ACM_FCM_paper_analytical(params["X"], 10)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x[exnersim.h.shape[0]//2], h[exnersim.h.shape[0]//2], linestyle="dashed", linewidth=3, c="black", label="Analytical")
    ax.plot(x[exnersim.h.shape[0]//2], h_init[exnersim.h.shape[0]//2], linestyle="dotted", linewidth=2, c="gray", label="Initial")
    ax.plot(x[exnersim.h.shape[0]//2], exnersim.h[exnersim.h.shape[0]//2], c="red", label="Numerical")
    ax.grid(alpha=0.25)
    ax.set_ylabel(r"Water depth $h$ [m]", fontsize=14)
    ax.set_xlabel(r"Channel Length $x$ [m]", fontsize=14)
    fig.legend(bbox_to_anchor=anchor, fontsize=12)
    fig.savefig("h_evolved.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x[exnersim.hu.shape[0]//2], q[exnersim.hu.shape[0]//2], linestyle="dashed", linewidth=3, c="black", label="Analytical")
    ax.plot(x[exnersim.hu.shape[0]//2], q_init[exnersim.hu.shape[0]//2], linestyle="dotted", linewidth=2, c="gray", label="Initial")
    ax.plot(x[exnersim.hu.shape[0]//2], exnersim.hu[exnersim.hu.shape[0]//2], c="red", label="Numerical")
    ax.grid(alpha=0.25)
    ax.set_ylim(2.9,3.02)
    ax.set_ylabel(r"Discharge $hu$ [$\mathrm{m}^2/\mathrm{s}$]", fontsize=14)
    ax.set_xlabel(r"Channel Length $x$ [m]", fontsize=14)
    fig.legend(bbox_to_anchor=(0.9,0.3), fontsize=12)
    fig.savefig("hu_evolved.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x[exnersim.z.shape[0]//2], zt[exnersim.z.shape[0]//2], linestyle="dashed", linewidth=3, c="black", label="Analytical")
    ax.plot(x[exnersim.z.shape[0]//2], zt_init[exnersim.z.shape[0]//2], linestyle="dotted", linewidth=2, c="gray", label="Initial")
    ax.plot(x[exnersim.z.shape[0]//2], exnersim.z[exnersim.z.shape[0]//2], c="red", label="Numerical")
    ax.grid(alpha=0.25)
    ax.set_ylabel(r"Bed height $z$ [m]", fontsize=14)
    ax.set_xlabel(r"Channel Length $x$ [m]", fontsize=14)
    fig.legend(bbox_to_anchor=(0.9,0.3), fontsize=12)
    fig.savefig("z_evolved.png", dpi=200)
    plt.close()

    
    dhs = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    l1_norm_z = [] 
    l2_norm_z = [] 
    linf_norm_z = [] 

    l1_norm_h = [] 
    l2_norm_h = [] 
    linf_norm_h = [] 

    for dh in dhs:
        print(dh)

        params = case_builder.ideal_case_ACM_FCM_paper_builder(T=10, dh=dh)
        zt, h, x, q = case_builder.ideal_case_ACM_FCM_paper_analytical(params["X"], t=10, dh=dh)
        
        exnersim = SWEExnerSim(params)
        exnersim.evolve()

        z_error = zt - exnersim.z

        l1_norm_z.append(dh*dh*np.sum(np.abs(z_error)))
        linf_norm_z.append(np.max(np.abs(z_error)))

        h_error = h - exnersim.h

        l1_norm_h.append(dh*dh*np.sum(np.abs(h_error)))
        linf_norm_h.append(np.max(np.abs(h_error)))  
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(l1_norm_h,   linestyle="dotted", marker=".", label=r"$L_1$ norm $h$")
    ax.plot(l1_norm_z, linestyle="dotted", marker=".", label=r"$L_1$ norm $z_b$")
    ax.grid(alpha=0.25, which="both")
    ax.set_ylabel(r"$L_1$ error [m]", fontsize=14)
    ax.set_xlabel(r"$\Delta x$ [m]", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks([i for i in range(len(dhs))])
    ax.set_xticklabels([f"${dh}$" for dh in dhs], fontsize=12)
    fig.legend(bbox_to_anchor=(0.9,0.825), fontsize=12)
    fig.savefig("L1_norms.png", dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(linf_norm_h, linestyle="dotted", marker=".", label=r"$L_\infty$ norm $h$")
    ax.plot(linf_norm_z, linestyle="dotted", marker=".", label=r"$L_\infty$ norm $z_b$")
    ax.grid(alpha=0.25,which="both")
    ax.set_ylabel(r"$L_\infty$ error [m]", fontsize=14)
    ax.set_xlabel(r"$\Delta x$ [m]", fontsize=14)
    ax.set_yscale("log")
    ax.set_xticks([i for i in range(len(dhs))])
    ax.set_xticklabels([f"${dh}$" for dh in dhs], fontsize=12)
    fig.legend(bbox_to_anchor=(0.9,0.825), fontsize=12)
    fig.savefig("Linf_norms.png", dpi=200)
    plt.close()
    """
