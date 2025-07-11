import numpy as np

# Constants
g = 9.81 # Gravity
TOL_WETDRY = 1e-12

def _get_theta(_lambda, utilde, ctilde):
    return 3*_lambda**2 - 4*utilde*_lambda + utilde**2 - ctilde**2

def _get_approx_lambda(_lambda, atilde, utilde, ctilde):
    theta = _get_theta(_lambda, utilde, ctilde)

    numerator = _lambda * theta - ctilde**2 * atilde * utilde 
    denominator = theta - ctilde**2*atilde 
    denominator = np.where(denominator < 1e-4, ctilde**2*atilde, denominator)  

    return numerator / denominator 

def compute_dt_SWE(h,hu,hv,dx):
    dt_x = np.min(dx / (np.abs(hu/h) + np.sqrt(g*h)))
    dt_y = np.min(dx / (np.abs(hv/h) + np.sqrt(g*h)))
    
    dt = np.minimum(dt_x, dt_y)
    return dt

def compute_dt(si, sj, nx, ny, dx):
    hi, hui, hvi, zi, gi = si
    hj, huj, hvj, zj, gj = sj
    
    sqrt_i = np.sqrt(hi)
    sqrt_j = np.sqrt(hj)

    ui = np.where(hi > 0.0, hui/hi, 0.0)
    vi = np.where(hi > 0.0, hvi/hi, 0.0)

    uj = np.where(hj > 0.0, huj/hj, 0.0)
    vj = np.where(hj > 0.0, hvj/hj, 0.0)

    uhati = ui*nx + vi*ny          # normal 
    vhati = -ui*ny + vi*nx         # tangent
    
    uhatj = uj*nx + vj*ny          # normal
    vhatj = -uj*ny + vj*nx         # tangent

    htilde = 0.5*(hi + hj)
    utilde = (uhati*sqrt_i + uhatj*sqrt_j)/(sqrt_i + sqrt_j)
    vtilde = (vhati*sqrt_i + vhatj*sqrt_j)/(sqrt_i + sqrt_j)
    ctilde = np.sqrt(g*htilde)
    gtilde = 0.5 * (gi + gj)
    atilde = gtilde*(uhati**2 + uhati*uhatj + uhatj**2 + vhati*vhatj)/(np.sqrt(hi*hj)) 
    
    # calculate wave speeds 

    lambda_1 = utilde - ctilde
    lambda_3 = utilde + ctilde

    lambda_approx_1 = _get_approx_lambda(lambda_1, atilde, utilde, ctilde)
    lambda_approx_4 = _get_approx_lambda(lambda_3, atilde, utilde, ctilde)

    dt = dx / np.maximum(np.abs(lambda_approx_1), np.abs(lambda_approx_4))
    dt = np.min(dt)

    return dt

def compute_dt_2D(h, hu, hv, z, G, dx):
    hi_x =   h[:,:-1]
    hui_x = hu[:,:-1]
    hvi_x = hv[:,:-1]
    zi_x =   z[:,:-1]
    gi_x =   G[:,:-1]

    hj_x =   h[:, 1:]
    huj_x = hu[:, 1:]
    hvj_x = hv[:, 1:]
    zj_x =   z[:, 1:]
    gj_x =   G[:, 1:]

    hi_y =   h[:-1,:]
    hui_y = hu[:-1,:]
    hvi_y = hv[:-1,:]
    zi_y =   z[:-1,:]
    gi_y =   G[:-1,:]

    hj_y =   h[1:, :]
    huj_y = hu[1:, :]
    hvj_y = hv[1:, :]
    zj_y =   z[1:, :]
    gj_y =   G[1:, :]
    
    s1_x = np.stack([
        hi_x,hui_x,hvi_x,zi_x,gi_x
    ])

    s2_x = np.stack([
        hj_x,huj_x,hvj_x,zj_x,gj_x
    ])
    
    s1_y = np.stack([
        hi_y,hui_y,hvi_y,zi_y,gi_y
    ])

    s2_y = np.stack([
        hj_y,huj_y,hvj_y,zj_y,gj_y
    ])

    dt_x = compute_dt(s1_x, s2_x, 1, 0, dx)
    dt_y = compute_dt(s1_y, s2_y, 0, -1, dx)

    dt = np.minimum(dt_x, dt_y)

    return dt

def roe_solver(si, sj, nx, ny, dx):

    # Unpack state variables for clarity
    hi, hui, hvi, zi, ni = si
    hj, huj, hvj, zj, nj = sj

    # Small epsilon to prevent division by zero or very small numbers
    epsilon = 1e-12

    hi = np.where(hi > epsilon, hi, 0.0)
    hj = np.where(hj > epsilon, hj, 0.0)

    sqrt_i = np.sqrt(hi)
    sqrt_j = np.sqrt(hj)

    # Calculate velocities, handling dry beds
    ui = np.where(hi > 0.0, hui/hi, 0.0)
    vi = np.where(hi > 0.0, hvi/hi, 0.0)

    uj = np.where(hj > 0.0, huj/hj, 0.0)
    vj = np.where(hj > 0.0, hvj/hj, 0.0)

    # --- Rotate problem to (normal, tangential) coordinate frame ---
    
    uhati = ui * nx + vi * ny   # Normal velocity component
    vhati = -ui * ny + vi * nx  # Tangential velocity component

    uhatj = uj * nx + vj * ny
    vhatj = -uj * ny + vj * nx

    # --- Roe Averages ---
    # To handle dry/wet transitions robustly, consider the 'h' average carefully.
    # The (sqrt_i + sqrt_j) in the denominator could be zero if both are dry.
    # Add an epsilon to the denominator or use a more robust average for dry states.

    htilde = np.where(0.5*(hi + hj) > epsilon, 0.5*(hi+hj), 0.0)
    ctilde = np.sqrt(g*htilde)

    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / (sqrt_i + sqrt_j)
    vtilde = (vhati * sqrt_i + vhatj * sqrt_j) / (sqrt_i + sqrt_j) 

    # --- Calculate Wave Speeds (Eigenvalues) ---
    lambda_1 = utilde - ctilde
    lambda_2 = utilde
    lambda_3 = utilde + ctilde

    # --- Right Eigenvector Matrix (P) in (h, hu_n, hv_t) basis ---
    # P[:, 0] is R_1 (left-going acoustic wave)
    # P[:, 1] is R_2 (contact discontinuity / shear wave)
    # P[:, 2] is R_3 (right-going acoustic wave)

    P = np.stack([
        np.stack([np.ones_like(vtilde), np.zeros_like(vtilde), np.ones_like(vtilde)],axis=-1),
        np.stack([lambda_1,             np.zeros_like(vtilde),             lambda_3],axis=-1),
        np.stack([vtilde,               ctilde,                              vtilde],axis=-1)
    ], axis=-2)

    # Construct the elements of the inner matrix (before factoring out 1/(2c))
    denom = lambda_1-lambda_3
    
    P_inv = np.stack([
        np.stack([-lambda_3/denom,      np.ones_like(utilde)/denom,     np.zeros_like(utilde)], axis=-1),
        np.stack([-vtilde/ctilde,       np.zeros_like(utilde),          np.ones_like(utilde)/ctilde], axis=-1),
        np.stack([lambda_1/denom,       -np.ones_like(utilde)/denom,    np.zeros_like(utilde)], axis=-1)
    ], axis=-2)

    # Calculate Inverse of P
    #P_inv = np.linalg.inv(P)

    # Entropy correction Harten-Hyman

    # lambda_1
    ei = uhati - np.sqrt(g*hi)
    ej = uhatj - np.sqrt(g*hj)

    mask_i = (ei < 0.0) & (ej > 0.0)

    lambda_1 = np.where(mask_i, ei*(ej-lambda_1)/((ej-ei)+1e-12), lambda_1) # Replace with Lambda hat
    lambda_E1 = np.where(mask_i, lambda_1 - ei*(ej-lambda_1)/((ej-ei)+1e-12), 0.0)
    
    # lambda_3
    ei = uhati + np.sqrt(g*hi)
    ej = uhatj + np.sqrt(g*hj)

    mask_j = (ei < 0.0) & (ej > 0.0)
    
    lambda_3 = np.where(mask_j, ej*(lambda_3-ei)/((ej-ei)+1e-12), lambda_3)      # Replace with Lambda hat
    lambda_E3 = np.where(mask_j, lambda_3 - ej*(lambda_3-ei)/((ej-ei)+1e-12), 0.0)

    lambda_E2 = np.zeros_like(lambda_E3) 
    lambdas_E = np.stack([lambda_E1, lambda_E2, lambda_E3], axis=-1)
    # Entropy correction Harten-Hyman

    lambdas = np.stack([lambda_1, lambda_2, lambda_3], axis=-1)

    # --- Difference in Conservative Variables (dU = Uj - Ui) ---
    dh = hj - hi
    # Note: dhu and dhv here are differences in normal and tangential momentum, not global.
    dhu = hj * uhatj - hi * uhati
    dhv = hj * vhatj - hi * vhati

    dU = np.stack([dh, dhu, dhv],axis=-1)

    # --- Wave Strengths (alphas) ---
    # alphas = P_inv * dU
    alphas = np.einsum("...ji,...i->...j", P_inv, dU)
    
    #alpha_1 = 0.5*(-lambda_1*dh - dhu)/ctilde 
    #alpha_2 = (dhv - vtilde*dh)/ctilde
    #alpha_3 = 0.5*(lambda_3*dh + dhu)/ctilde
    #alphas = np.stack([alpha_1, alpha_2, alpha_3], axis=-1)

    # --- Source Terms ---
    # Based on "http://dx.doi.org/10.1016/j.jcp.2010.02.016" 
    
    dz = zj - zi
    di = hi + zi
    dj = hj + zj
    dd = dj - di # Difference in total water depth

    # Topographic source term (pressure gradient due to bed slope)
    # This is a specific well-balanced formulation.
    thrust_a = -g * htilde * dz

    # Alternative thrust calculation for specific dry/wet conditions
    mask1_dz = (dz >= 0) & (di < zj)
    mask2_dz = (dz < 0) & (dj < zi)
    dztilde = np.where(mask1_dz, hi, np.where(mask2_dz, hj, dz))
    hr = np.where(dz >= 0, hi, hj) # Choose upstream or downstream h for reference height
    thrust_b = -g * (hr - 0.5 * np.abs(dztilde)) * dztilde

    # Combined thrust using specific conditions
    # This condition aims to correctly handle dry/wet fronts and supercritical/subcritical flow.
    # The condition (dz*dd >= 0.0) is often related to avoiding oscillations when the flow passes
    # over a hump or depression, and (utilde*dz > 0.0) when flow is moving uphill.
    mask_thrust = (dz * dd >= 0.0) & (utilde * dz > 0.0)
    thrust = np.where(mask_thrust, np.maximum(thrust_a, thrust_b), thrust_b)

    ntilde = 0.5*(ni+nj)
    sf = (ntilde**2*np.sqrt(utilde**2+vtilde**2)*utilde)/(np.maximum(epsilon, htilde**(4/3)))
    tau = g*htilde*sf*dx

    Tn = np.stack([
        np.zeros_like(htilde),
        thrust-tau,
        np.zeros_like(htilde)
    ], axis=-1)

    #beta_1 = -0.5*(thrust - tau)/ctilde
    #beta_2 = np.zeros_like(beta_1)
    #beta_3 = -beta_1.copy()
    betas = np.einsum("...ji,...i->...j", P_inv, Tn)
    #betas = np.stack([beta_1, beta_2, beta_3], axis=-1)

    # Reconstruction of approximate solution
    
    h_i1star = hi + alphas[...,0] - (betas[...,0]/lambdas[...,0])      # 1st intermediate state
    h_j3star = hj - alphas[...,2] + (betas[...,2]/lambdas[...,2])     # 3rd intermediate state
       
    beta1min = -(hi+alphas[...,0])*np.abs(lambdas[...,0])
    beta3min = -(hi-alphas[...,0])*np.abs(lambdas[...,2])

    dt = dx / np.max(np.abs(lambdas), axis=2)

    mask_1 = (h_i1star < 0.0) & (np.abs(hi) > 1e-12)
    mask_2 = (h_j3star < 0.0) & (np.abs(hj) > 1e-12)

    dt1star = (dx / 2*lambdas[...,0])*(hi/(hi-h_i1star + 1e-12)) 
    dt3star = (dx / 2*lambdas[...,2])*(hj/(hj-h_j3star + 1e-12))

    mask = (h_i1star < 0.0) & (h_j3star > 0.0) & (dt1star < dt)  
    betas[...,0] = np.where(mask, np.where(-beta1min >= beta3min, beta1min, betas[...,0]), betas[...,0])
    betas[...,2] = np.where(mask, -betas[...,0], betas[...,2]) 
    
    mask = (h_i1star > 0.0) & (h_j3star < 0.0) & (dt3star < dt)  
    betas[...,2] = np.where(mask, np.where(-beta3min >= beta3min, beta1min, betas[...,2]), betas[...,2])
    betas[...,0] = np.where(mask, -betas[...,2], betas[...,0])

    # Reconstruction of approximate solution
    upwP = np.zeros_like(lambdas)
    upwM = np.zeros_like(lambdas)

    mask_1 = (hi < epsilon) & (h_i1star < 0.0) 
    mask_2 = (hj < epsilon) & (h_j3star < 0.0)

    for i in range(lambdas.shape[-1]):
        flux = np.expand_dims(lambdas[...,i]*alphas[...,i]-betas[...,i],axis=2) * P[...,i]
        upwP += np.where(
            np.expand_dims(mask_2, axis=2), 0.0, np.where(
                np.expand_dims(mask_1,axis=2), flux, np.where(
                    np.expand_dims(lambdas[...,i], axis=2) > 0.0, flux, 0.0
                )
            )
        )

        upwM += np.where(
            np.expand_dims(mask_2,axis=2), flux, np.where(
                np.expand_dims(mask_1,axis=2), 0.0, np.where(
                    np.expand_dims(lambdas[...,i], axis=2) <= 0.0, flux, 0.0
                )
            )
        )

    for i in range(lambdas_E.shape[-1]):
        flux = np.expand_dims(lambdas_E[...,i]*alphas[...,i],axis=2) * P[...,i]
        upwP += np.where(np.expand_dims(lambdas_E[...,i], axis=2) > 0.0, flux, 0.0)
        upwM += np.where(np.expand_dims(lambdas_E[...,i], axis=2) <= 0.0, flux, 0.0)


    Tk_inv = np.array([[1,  0,   0],
                       [0, nx, -ny],
                       [0, ny,  nx]])

    upwP = np.einsum('ji,...i->...j', Tk_inv, upwP)
    upwM = np.einsum('ji,...i->...j', Tk_inv, upwM)

    return upwP, upwM


def roe_solve_2D(fluxes, h, hu, hv, z, n, dx):
    hi_x =   h[:,:-1]
    hui_x = hu[:,:-1]
    hvi_x = hv[:,:-1]
    zi_x =   z[:,:-1]
    ni_x =   n[:,:-1]

    hj_x =   h[:, 1:]
    huj_x = hu[:, 1:]
    hvj_x = hv[:, 1:]
    zj_x =   z[:, 1:]
    nj_x =   n[:, 1:]

    hi_y =   h[:-1,:]
    hui_y = hu[:-1,:]
    hvi_y = hv[:-1,:]
    zi_y =   z[:-1,:]
    ni_y =   n[:-1,:]

    hj_y =   h[1:, :]
    huj_y = hu[1:, :]
    hvj_y = hv[1:, :]
    zj_y =   z[1:, :]
    nj_y =   n[1:, :]
    
    s1_x = np.stack([
        hi_x,hui_x,hvi_x,zi_x,ni_x
    ])

    s2_x = np.stack([
        hj_x,huj_x,hvj_x,zj_x,nj_x
    ])
    
    s1_y = np.stack([
        hi_y,hui_y,hvi_y,zi_y,ni_y
    ])

    s2_y = np.stack([
        hj_y,huj_y,hvj_y,zj_y,nj_y
    ])

    upwP_x, upwM_x = roe_solver(s1_x, s2_x, 1,  0, dx)
    upwP_y, upwM_y = roe_solver(s1_y, s2_y, 0, -1, dx)

    # upwinding solution

    fluxes[:,:-1] += upwM_x 
    fluxes[:, 1:] += upwP_x 
    fluxes[:-1,:] += upwM_y 
    fluxes[1:, :] += upwP_y
    
    return fluxes
    
    
def exner_solve(s1, s2,  nx, ny, dx):
    # We will be implementing the Approximately Coupled Solver (ACM) https://doi.org/10.1016/j.advwatres.2021.103931
    hi, hui, hvi, zi, gi = s1
    hj, huj, hvj, zj, gj = s2
    sqrt_i = np.sqrt(hi)
    sqrt_j = np.sqrt(hj)

    # Calculate velocities, handling dry beds
    ui = np.where(hi > 1e-6, hui/hi, 0.0)
    vi = np.where(hi > 1e-6, hvi/hi, 0.0)

    uj = np.where(hj > 1e-6, huj/hj, 0.0)
    vj = np.where(hj > 1e-6, hvj/hj, 0.0)

    # --- Rotate problem to (normal, tangential) coordinate frame ---
    uhati = ui * nx + vi * ny   # Normal velocity component
    uhatj = uj * nx + vj * ny

    #qbnhati = qb_xi * nx + qb_yi * ny
    #qbnhatj = qb_xj * nx + qb_yj * ny

    utilde = (uhati * sqrt_i + uhatj * sqrt_j) / (sqrt_i + sqrt_j)

    umagi = (ui**2+vi**2)
    umagj = (uj**2+vj**2)

    qb_nhati = gi*umagi*uhati
    qb_nhatj = gj*umagj*uhatj

    gtilde = 0.5*(gi+gj)
    dz = zj - zi + 1e-14

    # figure this out later
    #qbhatL = gtilde*(uhati**2+vhati**2)*uhati - gi*(uhati**2+vhati**2)*uhati
    dqbhat = gtilde*umagj*uhatj - gtilde*umagi*uhati
    #qbhatR = gj*(uhatj**2+vhatj**2)*uhatj - gtilde*(uhatj**2+vhatj**2)*uhatj
    #dqbhat = qbhatL + qbhat + qbhatR
    #dqbhat = qb_nhatj - qb_nhati

    lambda_4 = np.where(np.abs(dz) > 1e-8, dqbhat / dz, utilde)

    corrector_i = (gtilde - gi)*umagi*uhati
    corrector_j = (gtilde - gj)*umagj*uhatj

    qbni = qb_nhati + corrector_i
    qbnj = qb_nhatj + corrector_j

    F_exner = np.where(lambda_4 >= 0, qbni, qbnj)

    return F_exner

def exner_solve_2D(fluxes, h, hu, hv, z, G, dx):
    hi_x =   h[:,:-1]
    hui_x = hu[:,:-1]
    hvi_x = hv[:,:-1]
    zi_x =   z[:,:-1]
    gi_x =   G[:,:-1]
    #qb_xi_x =   qb_x[:,:-1]
    #qb_yi_x =   qb_y[:,:-1]

    hj_x =   h[:, 1:]
    huj_x = hu[:, 1:]
    hvj_x = hv[:, 1:]
    zj_x =   z[:, 1:]
    gj_x =   G[:, 1:]
    #qb_xj_x = qb_x[:,1:]
    #qb_yj_x = qb_y[:,1:]

    hi_y =   h[:-1,:]
    hui_y = hu[:-1,:]
    hvi_y = hv[:-1,:]
    zi_y =   z[:-1,:]
    gi_y =   G[:-1,:]
    #qb_xi_y = qb_x[:-1,:]
    #qb_yi_y = qb_y[:-1,:]

    hj_y =   h[1:, :]
    huj_y = hu[1:, :]
    hvj_y = hv[1:, :]
    zj_y =   z[1:, :]
    gj_y =   G[1:, :]
    #qb_xj_y = qb_x[1:,:]
    #qb_yj_y = qb_y[1:,:]

    s1_x = np.stack([
        hi_x,hui_x,hvi_x,zi_x,gi_x #,qb_xi_x,qb_yi_x
    ])

    s2_x = np.stack([
        hj_x,huj_x,hvj_x,zj_x,gj_x #,qb_xj_x,qb_yj_x
    ])
    
    s1_y = np.stack([
        hi_y,hui_y,hvi_y,zi_y,gi_y #,qb_xi_y,qb_yi_y
    ])

    s2_y = np.stack([
        hj_y,huj_y,hvj_y,zj_y,gj_y #,qb_xj_y,qb_yj_y
    ])

    # Calculate the upwinded fluxes at the x-interfaces
    F_x = exner_solve(s1_x, s2_x, 1, 0, dx)

    fluxes[:, 1:] -= F_x            # F_x leaves cell to its right
    fluxes[:, :-1] += F_x           # F_x enters cell from its left 

    # Calculate the upwinded fluxes at the y-interfaces
    F_y = exner_solve(s1_y, s2_y, 0, -1, dx)

    fluxes[1:, :] += F_y # F_y enters cell from its bottom
    fluxes[:-1, :] -= F_y # F_y leaves cell to its top

    return fluxes

def wet_dry_correction(h, z, hu, hv, masks, hmin):
    ny, nx = h.shape

    isB = np.any(masks, axis=0)[1:-1, 1:-1]

    hij   = h[1:-1, 1:-1]
    zij   = z[1:-1, 1:-1]

    # Neighbor slices
    h_ip1 = h[1:-1, 2:]
    h_im1 = h[1:-1,:-2]
    h_jp1 = h[2:, 1:-1]
    h_jm1 = h[:-2,1:-1]

    z_ip1 = z[1:-1, 2:]
    z_im1 = z[1:-1,:-2]
    z_jp1 = z[2:, 1:-1]
    z_jm1 = z[:-2,1:-1]

    # Mask for h >= hmin
    wet_mask = hij >= hmin

    # X-direction condition
    cond_x = (
        ((hij + zij < z_ip1) & (h_ip1 < TOL_WETDRY)) |
        ((hij + zij < z_im1) & (h_im1 < TOL_WETDRY))
    ) & wet_mask 
     

    # Y-direction condition
    cond_y = (
        ((hij + zij < z_jp1) & (h_jp1 < TOL_WETDRY)) |
        ((hij + zij < z_jm1) & (h_jm1 < TOL_WETDRY))
    ) & wet_mask

    # Apply to hu, hv
    hu[1:-1, 1:-1][cond_x] = 0.0 
    hv[1:-1, 1:-1][cond_y] = 0.0

    return hu, hv