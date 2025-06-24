import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from roesolver import compute_dt_2D, roe_solve_2D, compute_dt_SWE, exner_solve_2D

@jax.jit
def point_in_polygon(point: jnp.ndarray, polygon: jnp.ndarray) -> bool:
    x, y = point[0], point[1]
    xi, yi = polygon[:, 0], polygon[:, 1]
    xj, yj = jnp.roll(xi, 1), jnp.roll(yi, 1)
    intersect = ((yi > y) != (yj > y)) & \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
    return jnp.sum(intersect) % 2 == 1

def compute_label_mask(points: jnp.ndarray, polygons: list[jnp.ndarray]) -> jnp.ndarray:
    masks_list = []
    for polygon in polygons:
        # jax.vmap applies point_in_polygon to each point in the 'points' batch
        # The 'polygon' argument remains static for this inner vmap.
        points_in_single_polygon = jax.vmap(point_in_polygon, in_axes=(0, None))(points, polygon)
        masks_list.append(points_in_single_polygon)
    
    return jnp.stack(masks_list)

class SWEExner:
    def __init__(self, params):
        self.params = params
        self._eps = 1e-12 

        # Initialize parameters as JAX arrays where appropriate
        self.endTime = params["endTime"]
        self.cfl = params["cfl"]
        self.outFreq = params["outFreq"]
        self.dx = params["dh"]
        self.cellArea = self.dx**2
        self.X = jnp.asarray(params["X"])
        self.Y = jnp.asarray(params["Y"])
        self.boundaries = params["boundaries"] # Keep as Python dict for now

        # Initial fields (converted to JAX arrays)
        self.h_init = jnp.asarray(params["h_init"])
        self.hu_init = jnp.asarray(params["h_init"] * params["u_init"])
        self.hv_init = jnp.asarray(params["h_init"] * params["v_init"])
        self.z_init = jnp.asarray(params["z_init"])
        self.G = jnp.asarray(params["A_g"]) # Renamed to avoid conflict with `G` in functions
        self.n = jnp.asarray(params["roughness"]) # Renamed to avoid conflict with `n` in functions

        # Pre-process boundary data
        self.polygons_jax = []
        self.boundary_values_jax = []
        self.normals_jax = []
        self.boundary_types = [] # To store string types for JIT-compatible switching

        for i, key in enumerate(self.boundaries.keys()):
            self.polygons_jax.append(jnp.asarray(self.boundaries[key]["polygon"]))
            self.boundary_values_jax.append(jnp.asarray(self.boundaries[key]["values"]))
            self.normals_jax.append(jnp.asarray(self.boundaries[key]["normal"]))
            self.boundary_types.append(self.boundaries[key]["type"])

        # Pad polygons to same length for jax.vmap
        max_vertices = max(p.shape[0] for p in self.polygons_jax)
        self.padded_polygons_jax = jnp.stack([
            jnp.pad(p, ((0, max_vertices - p.shape[0]), (0, 0))) for p in self.polygons_jax
        ])
        
        points_flat = jnp.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        # Use the fully JITted mask computation
        self.masks_flat = compute_label_mask_fully_vmap(points_flat, self.padded_polygons_jax)
        self.masks = self.masks_flat.reshape(len(self.polygons_jax), self.X.shape[0], self.X.shape[1])
        
        # JIT-compile the apply_boundaries function once
        # We need to map string boundary types to integers for jax.lax.switch
        self._type_map = {
            "constant_flux": 0,
            "normal_flow_depth": 1,
            "transmissive_bedload": 2,
            "transmissive_flux": 3,
            "transmissive_bounds": 4
        }
        self._apply_boundaries_jitted = jax.jit(self._apply_boundaries_core)
        

    def _initialize_sim_state(self):
        # Initial simulation state
        # These will be updated in the simulation loop, so they need to be part of the state tuple
        h = self.h_init
        hu = self.hu_init
        hv = self.hv_init
        z = self.z_init
        qb_x = self.G_val * ((hu/h)**2 + (hv/h)**2) * (hu/h)
        qb_y = self.G_val * ((hu/h)**2 + (hv/h)**2) * (hv/h)
        contributions = jnp.zeros((h.shape[0], h.shape[1], 3))
        contributions_z = jnp.zeros_like(z)
        etime = 0.0
        dt = 1.0 # Initial dummy dt
        
        # Apply initial boundaries
        h, hu, hv, z, qb_x, qb_y = self._apply_boundaries_jitted(
            h, hu, hv, z, qb_x, qb_y,
            self.masks,
            self.boundary_values_jax,
            self.normals_jax,
            jnp.array([self._type_map[t] for t in self.boundary_types]), # Pass int codes
            self.G # Pass G explicitly
        )
        
        return (h, hu, hv, z, qb_x, qb_y, contributions, contributions_z, etime, dt, 0) # iter_count

    # JIT-compiled core function for applying boundaries
    # This must handle all boundary types using JAX-compatible control flow (e.g., lax.switch)
    @partial(jax.jit, static_argnums=(8,)) # boundary_types_code is dynamic; we pass the type map indices.
    def _apply_boundaries_core(h, hu, hv, z, qb_x, qb_y, masks, boundary_values, normals, boundary_types_code, G):
        
        def apply_constant_flux(h, hu, hv, z, qb_x, qb_y, mask, boundary_value, normal, G):
            qx = boundary_value[1] * normal[0]
            qy = boundary_value[1] * normal[1]
            hu = hu.at[mask].set(qx)
            hv = hv.at[mask].set(qy)
            u = jnp.where(h[mask] != 0, qx / h[mask], 0.0) # Avoid division by zero
            v = jnp.where(h[mask] != 0, qy / h[mask], 0.0)
            qb_x = qb_x.at[mask].set(boundary_value[3]) # Assuming boundary_value[3] is the qb_x value
            qb_y = qb_y.at[mask].set(boundary_value[3]) # Assuming boundary_value[3] is the qb_y value
            return h, hu, hv, z, qb_x, qb_y

        def apply_normal_flow_depth(h, hu, hv, z, qb_x, qb_y, mask, boundary_value, normal, G):
            h_val = boundary_value[0]
            qx = boundary_value[1] * normal[0]
            qy = boundary_value[1] * normal[1]
            h = h.at[mask].set(h_val)
            hu = hu.at[mask].set(qx)
            hv = hv.at[mask].set(qy)
            u = jnp.where(h_val != 0, qx / h_val, 0.0)
            v = jnp.where(h_val != 0, qy / h_val, 0.0)
            qb_x = qb_x.at[mask].set(G * (u**2 + v**2) * u)
            qb_y = qb_y.at[mask].set(G * (u**2 + v**2) * v)
            return h, hu, hv, z, qb_x, qb_y

        def apply_transmissive_bedload(h, hu, hv, z, qb_x, qb_y, mask, boundary_value, normal, G):
            shift = -jnp.rint(jnp.flip(normal)).astype(int)
            boundary_cells = jnp.argwhere(mask)
            interior_cells = boundary_cells + shift
            Ny, Nx = mask.shape
            interior_cells = jnp.clip(interior_cells, a_min=jnp.array([0, 0]), a_max=jnp.array([Ny - 1, Nx - 1]))
            
            boundary_y, boundary_x = boundary_cells[:, 0], boundary_cells[:, 1]
            interior_y, interior_x = interior_cells[:, 0], interior_cells[:, 1]

            qb_x = qb_x.at[boundary_y, boundary_x].set(qb_x[interior_y, interior_x])
            qb_y = qb_y.at[boundary_y, boundary_x].set(qb_y[interior_y, interior_x])
            z = z.at[boundary_y, boundary_x].set(z[interior_y, interior_x])
            return h, hu, hv, z, qb_x, qb_y

        def apply_transmissive_flux(h, hu, hv, z, qb_x, qb_y, mask, boundary_value, normal, G):
            shift = -jnp.rint(jnp.flip(normal)).astype(int)
            boundary_cells = jnp.argwhere(mask)
            interior_cells = boundary_cells + shift
            Ny, Nx = mask.shape
            interior_cells = jnp.clip(interior_cells, a_min=jnp.array([0, 0]), a_max=jnp.array([Ny - 1, Nx - 1]))
            
            boundary_y, boundary_x = boundary_cells[:, 0], boundary_cells[:, 1]
            interior_y, interior_x = interior_cells[:, 0], interior_cells[:, 1]

            h = h.at[boundary_y, boundary_x].set(h[interior_y, interior_x])
            hu = hu.at[boundary_y, boundary_x].set(hu[interior_y, interior_x])
            hv = hv.at[boundary_y, boundary_x].set(hv[interior_y, interior_x])
            return h, hu, hv, z, qb_x, qb_y

        def apply_transmissive_bounds(h, hu, hv, z, qb_x, qb_y, mask, boundary_value, normal, G):
            shift = -jnp.rint(jnp.flip(normal)).astype(int)
            boundary_cells = jnp.argwhere(mask)
            interior_cells = boundary_cells + shift
            Ny, Nx = mask.shape
            interior_cells = jnp.clip(interior_cells, a_min=jnp.array([0, 0]), a_max=jnp.array([Ny - 1, Nx - 1]))
            
            boundary_y, boundary_x = boundary_cells[:, 0], boundary_cells[:, 1]
            interior_y, interior_x = interior_cells[:, 0], interior_cells[:, 1]

            h = h.at[boundary_y, boundary_x].set(h[interior_y, interior_x])
            hu = hu.at[boundary_y, boundary_x].set(hu[interior_y, interior_x])
            hv = hv.at[boundary_y, boundary_x].set(hv[interior_y, interior_x])
            qb_x = qb_x.at[boundary_y, boundary_x].set(qb_x[interior_y, interior_x])
            qb_y = qb_y.at[boundary_y, boundary_x].set(qb_y[interior_y, interior_x])
            z = z.at[boundary_y, boundary_x].set(z[interior_y, interior_x])
            return h, hu, hv, z, qb_x, qb_y

        # Define an array of functions for lax.switch
        # Order must match the integer codes in _type_map
        apply_funcs = [
            apply_constant_flux,
            apply_normal_flow_depth,
            apply_transmissive_bedload,
            apply_transmissive_flux,
            apply_transmissive_bounds
        ]

        # Loop through each boundary and apply its specific logic
        # This loop will be unrolled by JIT if boundary_types_code is a static arg,
        # or it will be a dynamic loop if it's not (which we want for `jax.vmap` compatibility).
        # We use a scan here to make the loop itself JIT-compatible.
        def _single_boundary_scan_fn(carry, i):
            h, hu, hv, z, qb_x, qb_y = carry
            mask = masks[i]
            boundary_value = boundary_values[i]
            normal = normals[i]
            type_code = boundary_types_code[i]

            # Use lax.switch to select the correct boundary application function
            # This replaces the Python `match-case` statement.
            h, hu, hv, z, qb_x, qb_y = jax.lax.switch(
                type_code,
                apply_funcs,
                h, hu, hv, z, qb_x, qb_y, mask, boundary_value, normal, G
            )
            return (h, hu, hv, z, qb_x, qb_y), None # None for no scanned output

        # Apply boundaries sequentially using lax.scan
        # initial_carry = (h, hu, hv, z, qb_x, qb_y)
        (h, hu, hv, z, qb_x, qb_y), _ = jax.lax.scan(
            _single_boundary_scan_fn,
            (h, hu, hv, z, qb_x, qb_y),
            jnp.arange(len(boundary_types_code)) # Iterate over boundary indices
        )
        return h, hu, hv, z, qb_x, qb_y

    #@partial(jax.jit, static_argnums=(0,)) # `self` is static
    #def _update_sediment_fluxes(self, h, hu, hv, G):
    #    qb_x = G * ((hu / h)**2 + (hv / h)**2) * (hu / h)
    #    qb_y = G * ((hu / h)**2 + (hv / h)**2) * (hv / h)
    #    return qb_x, qb_y

    # This function defines one simulation step, which will be called within lax.while_loop
    @partial(jax.jit, static_argnums=(0,)) # `self` is static
    def _sim_step(self, sim_state):
        h, hu, hv, z, qb_x, qb_y, contributions, contributions_z, etime, dt, iters = sim_state

        dt = compute_dt_2D(h, hu, hv, z, self.G_val, self.dx) * self.cfl

        # Adjust dt for output frequency
        next_out_time = (jnp.floor(etime / self.outFreq) + 1) * self.outFreq
        next_target = jnp.minimum(next_out_time, self.endTime)
        dt = jnp.where(etime + dt >= next_target - self._eps, next_target - etime, dt)
        
        # Step hydro
        new_contributions = roe_solve_2D(contributions, h, hu, hv, z, self.n_val, self.dx)
        h_new = h - dt * new_contributions[..., 0] / self.dx
        hu_new = hu - dt * new_contributions[..., 1] / self.dx
        hv_new = hv - dt * new_contributions[..., 2] / self.dx
        
        # Update sediment fluxes (pure function)
        qb_x_new, qb_y_new = self._update_sediment_fluxes(h_new, hu_new, hv_new, self.G_val)

        # Apply boundaries (pure function, already JIT-compiled in __init__)
        h_final, hu_final, hv_final, z_final, qb_x_final, qb_y_final = self._apply_boundaries_jitted(
            h_new, hu_new, hv_new, z, qb_x_new, qb_y_new,
            self.masks,
            self.boundary_values_jax,
            self.normals_jax,
            jnp.array([self._type_map[t] for t in self.boundary_types]),
            self.G_val
        )

        # Step exner
        new_contributions_z = exner_solve_2D(contributions_z, h_final, hu_final, hv_final, z_final, self.G_val, self.dx)
        z_final = z_final - dt * new_contributions_z / self.dx

        etime_new = etime + dt
        iters_new = iters + 1
        
        # Reset contributions (they are recomputed at each step)
        contributions_new = jnp.zeros_like(contributions)
        contributions_z_new = jnp.zeros_like(contributions_z)

        return (h_final, hu_final, hv_final, z_final, qb_x_final, qb_y_final, 
                contributions_new, contributions_z_new, etime_new, dt, iters_new)

    def evolve(self):
        # Initial simulation state
        # (h, hu, hv, z, qb_x, qb_y, contributions, contributions_z, etime, dt, iters)
        sim_state = self._initialize_sim_state()

        # Define the loop condition for lax.while_loop
        def loop_cond(current_state):
            etime = current_state[8] # etime is the 9th element (index 8)
            return etime < self.endTime - self._eps

        # Define the loop body for lax.while_loop
        @partial(jax.jit, static_argnums=(0,)) # `self` is static
        def loop_body(current_state):
            h, hu, hv, z, qb_x, qb_y, contributions, contributions_z, etime, dt, iters = current_state
            
            next_state = self._sim_step(current_state)

            # Debugging and printing logic (outside the JITted loop)
            # This part needs to be handled carefully. lax.while_loop itself can't have side effects.
            # We can use jax.debug.print for debugging inside JIT (but won't save plots).
            # For saving plots, we either save the state at intervals and plot outside,
            # or use a `lax.scan` over predetermined time points to collect states.
            # For a simple `while_loop`, we'll just print final state or save periodically outside the loop.

            # You might return states for plotting or use jax.debug.print
            # jax.debug.print("Iteration: {iters} Time: {etime}", iters=next_state[10], etime=next_state[8])
            
            return next_state

        # Run the simulation loop
        final_state = jax.lax.while_loop(loop_cond, loop_body, sim_state)

        # Unpack final state
        self.h, self.hu, self.hv, self.z, self.qb_x, self.qb_y, _, _, self.etime, self.dt, self.iters = final_state

        print(f"Simulation finished at Time: {self.etime:.9f} after {self.iters} iterations.")
        
        # Post-simulation debugging/plotting
        self._debug_boundaries(self.iters) # Plot final state
        
    def _debug_boundaries(self, i):
        # This function uses matplotlib and print, so it cannot be JIT-compiled.
        # It operates on the class's state after the JITted evolve function finishes or during a manual debug call.
        os.makedirs("debug", exist_ok=True) # Ensure debug folder exists

        # Convert JAX arrays to NumPy for matplotlib
        qb_x_np = jnp.asarray(self.qb_x)
        qb_y_np = jnp.asarray(self.qb_y)
        hu_np = jnp.asarray(self.hu)
        hv_np = jnp.asarray(self.hv)
        h_np = jnp.asarray(self.h)
        z_np = jnp.asarray(self.z)
        X_np = jnp.asarray(self.X)

        etime_str = f"{self.etime:.3f}"

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(qb_x_np)
        ax[1].plot(X_np[X_np.shape[0]//2,:], qb_x_np[qb_x_np.shape[0]//2,:])
        fig.suptitle(f"Time: {etime_str}")
        ax[1].grid(alpha=0.25)
        fig.savefig(f"debug/debug_qb_x{i}.png")
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(qb_y_np)
        ax[1].plot(X_np[X_np.shape[0]//2,:], qb_y_np[qb_y_np.shape[0]//2,:])
        fig.suptitle(f"Time: {etime_str}")
        ax[1].grid(alpha=0.25)
        fig.savefig(f"debug/debug_qb_y{i}.png")
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(hu_np)
        ax[1].plot(X_np[X_np.shape[0]//2,:], hu_np[hu_np.shape[0]//2,:])
        fig.suptitle(f"Time: {etime_str}")
        ax[1].grid(alpha=0.25)
        fig.savefig(f"debug/debug_hu{i}.png")
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(hv_np)
        ax[1].plot(X_np[X_np.shape[0]//2,:], hv_np[hv_np.shape[0]//2,:])
        ax[1].grid(alpha=0.25)
        fig.suptitle(f"Time: {etime_str}")
        fig.savefig(f"debug/debug_hv{i}.png")
        plt.close()
        
        fig, ax = plt.subplots(2,1,figsize=(12,5))
        ax[0].imshow(h_np)
        ax[1].plot(X_np[X_np.shape[0]//2,:], h_np[h_np.shape[0]//2,:]+z_np[z_np.shape[0]//2,:],label="h+z")
        ax[1].plot(X_np[X_np.shape[0]//2,:], z_np[z_np.shape[0]//2,:],label="z")
        ax[1].grid(alpha=0.25)
        fig.suptitle(f"Time: {etime_str}")
        ax[1].legend()
        ax[1].set_xlim(-2.75, 2.75)
        ax[1].set_ylim(1.18, 1.8)
        fig.savefig(f"debug/debug_h+z{i}.png")
        plt.close()

        fig, ax = plt.subplots(2,1,figsize=(12,6.5),sharex=True)
        ax[0].plot(X_np[X_np.shape[0]//2,:], h_np[h_np.shape[0]//2,:]+z_np[z_np.shape[0]//2,:],c="blue",
                    marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax[0].grid(alpha=0.25)
        ax[0].set_xlim(-2.75, 2.75)
        ax[0].set_ylim(1.18, 1.8)
        fig.suptitle(f"Time: {etime_str}")
        ax[0].set_ylabel("Free Surf level (m)")
        ax[1].plot(X_np[X_np.shape[0]//2,:], z_np[z_np.shape[0]//2,:],c="blue",
                    marker=".",linewidth=1,markerfacecolor='none', markeredgecolor="blue")
        ax[1].grid(alpha=0.25)
        ax[1].set_xlim(-2.75, 2.75)
        ax[1].set_ylim(0.9, 1.05)
        ax[1].set_ylabel("Bed level (m)")
        fig.suptitle(f"Time: {etime_str}")
        fig.savefig(f"debug/debug_hz{i:06d}.png")
        plt.close()


# Example Usage:
if __name__ == "__main__":
    # Dummy parameters for testing
    grid_size = 50
    params = {
        "endTime": 1.0,
        "cfl": 0.5,
        "outFreq": 0.5,
        "h_init": jnp.ones((grid_size, grid_size)) * 1.0,
        "u_init": jnp.zeros((grid_size, grid_size)),
        "v_init": jnp.zeros((grid_size, grid_size)),
        "z_init": jnp.ones((grid_size, grid_size)) * 0.0,
        "A_g": 0.1, # G
        "roughness": 0.03, # n
        "dh": 0.1, # dx
        "X": jnp.linspace(-2.5, 2.5, grid_size),
        "Y": jnp.linspace(-2.5, 2.5, grid_size).reshape(-1, 1),
        "boundaries": {
            "inlet": {
                "polygon": jnp.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]),
                "values": jnp.array([1.2, 0.5, 0.0, 0.0]), # h, q, ...
                "normal": jnp.array([1.0, 0.0]),
                "type": "constant_flux"
            },
            "outlet": {
                "polygon": jnp.array([[-0.1, -0.1], [0.0, -0.1], [0.0, 0.0], [-0.1, 0.0]]),
                "values": jnp.array([0.0, 0.0, 0.0, 0.0]),
                "normal": jnp.array([-1.0, 0.0]),
                "type": "transmissive_bounds"
            }
        }
    }

    # Initialize the simulator
    sim = SWEExnerSimJax(params)

    # Evolve the simulation
    sim.evolve()