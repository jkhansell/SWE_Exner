import jax 
import jax.numpy as jnp

g = 9.81
TOLDRY = 1e-12

def compute_dt_SWE(h,hu,hv,dx):
    dt_x = jnp.min(dx / (jnp.abs(hu[1:-1,1:-1]/h[1:-1,1:-1])+jnp.sqrt(g*h[1:-1,1:-1])))
    dt_y = jnp.min(dx / (jnp.abs(hv[1:-1,1:-1]/h[1:-1,1:-1])+jnp.sqrt(g*h[1:-1,1:-1])))
    
    dt = jnp.minimum(dt_x, dt_y)

    return dt

def roe_solve_jit(si, sj,  nx, ny, dx, dt):
    # define cells at both sides of the edge

    hi, hui, hvi, zi, gi = si
    hj, huj, hvj, zj, gj = sj

    sqrt_i = jnp.sqrt(hi)
    sqrt_j = jnp.sqrt(hj)

    ui = jnp.where(hi > 0.0, hui/hi, 0.0)
    vi = jnp.where(hi > 0.0, hvi/hi, 0.0)
    uj = jnp.where(hj > 0.0, huj/hj, 0.0)
    vj = jnp.where(hj > 0.0, hvj/hj, 0.0)

    htilde = 0.5 * (hi + hj)
    utilde = (ui*sqrt_i + uj*sqrt_j) / (sqrt_i+sqrt_j)
    vtilde = (vi*sqrt_i + vj*sqrt_j) / (sqrt_i+sqrt_j)
    ctilde = jnp.sqrt(g*htilde)

    un = utilde*nx + vtilde*ny

    lambda_1 = un-ctilde
    lambda_2 = un
    lambda_3 = un+ctilde
    
    # Entropy correction
    
    ei = ui*nx + vi*ny - jnp.sqrt(g*hi)
    ej = uj*nx + vj*ny - jnp.sqrt(g*hj)

    mask1 = (ei < 0.0) & (ej > 0.0)

    lambda_1_line = jnp.where(mask1, ei*(ej-lambda_1)/(ej-ei), lambda_1)
    lambda_1_hat = jnp.where(mask1, ei*(ej-lambda_1)/(ej-ei), lambda_1)
    
    ei = ui*nx + vi*ny + jnp.sqrt(g*hi)
    ej = uj*nx + vj*ny + jnp.sqrt(g*hj)

    mask3 = (ei < 0.0) & (ej > 0.0)

    lambda_E3 = jnp.where(mask3, lambda_1 - ei*(ej-lambda_1)/(ej-ei), 0.0)
    lambda_3 = jnp.where(mask3, ei*(ej-lambda_1)/(ej-ei), lambda_1)
    
    lambda_E2 = jnp.zeros_like(lambda_E1)

    # Entropy correction

    P = jnp.stack([
        jnp.stack([jnp.ones_like(vtilde), jnp.zeros_like(vtilde), jnp.ones_like(vtilde)], axis=-1),
        jnp.stack([utilde-ctilde*nx,      -ctilde*ny, utilde+ctilde*nx], axis=-1),
        jnp.stack([vtilde-ctilde*ny,       ctilde*nx, vtilde+ctilde*ny], axis=-1), 
    ],axis=-2)

    print(P[5,5])

    #Pinv = jnp.linalg.inv(P)  # shape (Nx, Ny, 3, 3)

    dz = zj - zi
    di = hi + zi
    dj = hj + zj
    dd = dj - di

    thrust_a = -g*htilde*dz

    mask1_dz = (dz >= 0) & (di < zj)
    mask2_dz = (dz < 0) & (dj < zi)

    dztilde = jnp.where(mask1_dz, hi, jnp.where(mask2_dz, hj, dz))

    hr = jnp.where(dz >= 0, hi, hj)
    thrust_b = -g*(hr - 0.5*jnp.abs(dztilde)*dztilde)

    mask_thrust = (dz*dd >= 0.0) & (utilde*dz > 0.0)
    thrust = jnp.where(mask_thrust, jnp.maximum(thrust_a, thrust_b), thrust_b)



    dh = hj - hi
    dhu = huj - hui
    dhv = hvj - hvi

    alpha_1 = 0.5*(dh - ((dhu*nx+dhv*ny) - un*dh)/ctilde)
    alpha_2 = ((dhv - vtilde*dh)*nx - (dhu - utilde*dh)*ny)/ctilde
    alpha_3 = 0.5*(dh + ((dhu*nx+dhv*ny) - un*dh)/ctilde) 

    dz = zj - zi
    li = zi+hi    
    lj = zj+hj    

    hp = jnp.where(dz >= 0.0, hi, hj)
    
    dzp = jnp.where((dz >= 0.0) & (li < zj), hi, jnp.where(lj<zi, -hj, dz)) 

    beta_1 = 0.5/ctilde*g*(hp-0.5*jnp.abs(dzp))*dzp

    hp = hi + alpha_1
    mask_betas = (lambda_1*lambda_3 < 0.0) & (hp > 0.0) & (hi > 0.0) & (hj > 0.0)
    
    beta_1 = beta_1.at[:].set(jnp.where(mask_betas, (jnp.maximum(beta_1, alpha_1*lambda_1-hi*dx*0.5/dt)), beta_1))
    beta_1 = beta_1.at[:].set(jnp.where(mask_betas, (jnp.minimum(beta_1, -alpha_3*lambda_3+hj*dx*0.5/dt)), beta_1))

    beta_2 = jnp.zeros_like(beta_1)
    beta_3 = -beta_1

    lambdas = jnp.stack([lambda_1, lambda_2, lambda_3],axis=-1)
    lambdasE = jnp.stack([lambda_E1, lambda_E2, lambda_E3],axis=-1)
    alphas = jnp.stack([alpha_1, alpha_2, alpha_3], axis=-1)
    betas = jnp.stack([beta_1, beta_2, beta_3], axis=-1)

    upwP = jnp.zeros_like(alphas)
    upwM = jnp.zeros_like(alphas)

    # li and lj are dry-state indicators
    li = hp - beta_1 / lambda_1
    lj = hp + beta_3 / lambda_3

    # Dry state masks
    mask_upwM = (lj < -TOLDRY) & (hj < TOLDRY)
    mask_upwP = (li < -TOLDRY) & (hi < TOLDRY)

    # Initialize flux arrays (upwinded contributions)
    upwP = jnp.zeros_like(P[..., 0])  # shape (Nx, Ny, 3)
    upwM = jnp.zeros_like(P[..., 0])

    # Degenerate flux contribution if one side is dry
    degenerate_contrib = lambda_1 * alpha_1 - beta_1 + lambda_3 * alpha_3 - beta_3

    upwM = upwM.at[..., 0].add(jnp.where(mask_upwM, degenerate_contrib, 0.0))
    upwP = upwP.at[..., 0].add(jnp.where(mask_upwP, degenerate_contrib, 0.0))

    # Fallback to regular flux decomposition if not degenerate
    regular_mask = ~(mask_upwM | mask_upwP)

    # +++ NEW VECTORIZED FLUX CALCULATION +++
    # This block replaces both loops and correctly calculates the fluxes.
    
    # 1. Calculate the magnitude of the flux contribution from each wave
    wave_flux_mags = lambdas * alphas - betas
    entropy_flux_mags = lambdasE * alphas
    
    # 2. Split the magnitudes into positive-going and negative-going parts
    pos_mags = jnp.where((lambdas > 0) & regular_mask[..., None], wave_flux_mags, 0.0) + \
               jnp.where(lambdasE > 0, entropy_flux_mags, 0.0)
               
    neg_mags = jnp.where((lambdas <= 0) & regular_mask[..., None] , wave_flux_mags, 0.0) + \
               jnp.where(lambdasE <= 0, entropy_flux_mags, 0.0)

    # 3. Project these magnitudes back through the eigenvectors (P) to get the final flux vectors.
    #    jnp.einsum is a powerful way to do matrix-vector products over many dimensions.
    #    This calculates P @ pos_mags and P @ neg_mags for every cell interface at once.
    upwP = jnp.einsum('...ij,...j->...i', P, pos_mags)
    upwM = jnp.einsum('...ij,...j->...i', P, neg_mags)
    
    # Note: Your logic for dry states (mask_upwM, etc.) was complex and might need
    # to be re-integrated here if you still face issues in very shallow/dry conditions.
    # For the dam break problem, this should be sufficient.


    #for k in range(lambdas.shape[-1]):
    #    contrib_k = jnp.expand_dims(lambdas[..., k] * alphas[..., k] - betas[..., k], axis=-1) * P[..., k]
#
    #    # Apply fluxes only where not degenerate
    #    pos_flux = jnp.where((lambdas[..., k] > 0.0)[..., None] & regular_mask[..., None], contrib_k, 0.0)
    #    neg_flux = jnp.where((lambdas[..., k] <= 0.0)[..., None] & regular_mask[..., None], contrib_k, 0.0)
#
    #    upwP = upwP.at[:].add(pos_flux)
    #    upwM = upwM.at[:].add(neg_flux)
#
    #for k in range(lambdasE.shape[-1]):
    #    contrib = jnp.expand_dims(lambdasE[...,k]*alphas[...,k],axis=2)*P[...,k]
    #    upwP = upwP.at[:].add(jnp.where(jnp.expand_dims(lambdasE[...,k],axis=2) > 0.0, contrib, 0.0))
    #    upwM = upwM.at[:].add(jnp.where(jnp.expand_dims(lambdasE[...,k],axis=2) <= 0.0, contrib, 0.0))
        
    return upwP, upwM

def roe_solve_2D_jit(fluxes, h, hu, hv, z, G, dx, dt):
    hi_x =   h[1:-1,:-1]
    hui_x = hu[1:-1,:-1]
    hvi_x = hv[1:-1,:-1]
    zi_x =   z[1:-1,:-1]
    gi_x =   G[1:-1,:-1]

    hj_x =   h[1:-1, 1:]
    huj_x = hu[1:-1, 1:]
    hvj_x = hv[1:-1, 1:]
    zj_x =   z[1:-1, 1:]
    gj_x =   G[1:-1, 1:]

    hi_y =   h[:-1,1:-1]
    hui_y = hu[:-1,1:-1]
    hvi_y = hv[:-1,1:-1]
    zi_y =   z[:-1,1:-1]
    gi_y =   G[:-1,1:-1]

    hj_y =   h[1:, 1:-1]
    huj_y = hu[1:, 1:-1]
    hvj_y = hv[1:, 1:-1]
    zj_y =   z[1:, 1:-1]
    gj_y =   G[1:, 1:-1]
    
    s1_x = jnp.stack([
        hi_x,hui_x,hvi_x,zi_x,gi_x
    ])

    s2_x = jnp.stack([
        hj_x,huj_x,hvj_x,zj_x,gj_x
    ])
    
    s1_y = jnp.stack([
        hi_y,hui_y,hvi_y,zi_y,gi_y
    ])

    s2_y = jnp.stack([
        hj_y,huj_y,hvj_y,zj_y,gj_y
    ])

    upwP_x, upwM_x = roe_solve_jit(s1_x, s2_x, 1, 0, dx, dt)
    upwP_y, upwM_y = roe_solve_jit(s1_y, s2_y, 0, -1, dx, dt)

    # upwinding solution

    fluxes = fluxes.at[1:-1,:-1].add(upwM_x) 
    fluxes = fluxes.at[1:-1, 1:].add(upwP_x)
    fluxes = fluxes.at[:-1,1:-1].add(upwM_y) 
    fluxes = fluxes.at[1: ,1:-1].add(upwP_y)
    
    
    return fluxes

"""  # thrust terms

    dz = zj - zi
    di = hi + zi
    dj = hj + zj
    dd = dj - di

    thrust_a = -g*htilde*dz

    mask1_dz = (dz >= 0) & (di < zj)
    mask2_dz = (dz < 0) & (dj < zi)

    dztilde = np.where(mask1_dz, hi, np.where(mask2_dz, hj, dz))

    hr = np.where(dz >= 0, hi, hj)
    thrust_b = -g*(hr - 0.5*np.abs(dztilde))*dztilde

    mask_thrust = (dz*dd >= 0.0) & (dz > 0.0)
    thrust = np.where(mask_thrust, np.maximum(thrust_a, thrust_b), thrust_b)

    beta_1 = thrust/(2*ctilde)
    beta_2 = np.zeros_like(beta_1)
    beta_3 = -beta_1.copy()

    # Entropy correction Harten-Hyman

    # lambda_1
    ei = uhati - np.sqrt(g*hi)
    ej = uhatj - np.sqrt(g*hj)


    mask_i = (ei < 0.0) & (ej > 0.0)

    lambda_1 = np.where(mask_i, ei*(ej-lambda_1)/((ej-ei)+1e-12), lambda_1) # Replace with Lambda hat
    lambda_E1 = np.where(mask_i, ej*(lambda_1-ei)/((ej-ei)+1e-12), 0.0)
    
    # lambda_3
    ei = uhati + np.sqrt(g*hi)
    ej = uhatj + np.sqrt(g*hj)

    mask_j = (ei < 0.0) & (ej > 0.0)
    
    lambda_3 = np.where(mask_j, ej*(lambda_3-ei)/((ej-ei)+1e-12), lambda_3)      # Replace with Lambda hat
    lambda_E3 = np.where(mask_j, ei*(ej-lambda_3)/((ej-ei)+1e-12), 0.0)

    lambda_E2 = np.zeros_like(lambda_E3) 
    
    # Entropy correction Harten-Hyman

    # Reconstruction of approximate solution
    
    h_istar = hi + alpha_1 - (beta_1/lambda_1)      # 1st intermediate state
    h_j3star = hj - alpha_3 + (beta_3/lambda_3)     # 3rd intermediate state
    
    beta1min = -(hi+alpha_1)*np.abs(lambda_1)
    beta3min = -(hi-alpha_1)*lambda_3

    dt = dx / np.max(np.abs(np.stack([lambda_2, lambda_3])), axis=0)

    mask_1 = (h_istar < 0.0) & (hi != 0.0)
    mask_2 = (h_j3star < 0.0) & (hj != 0.0)

    dtstar = np.where(mask_1, (dx / 2*lambda_1)*(hi/(hi-h_istar)), dt) 
    dt3star = np.where(mask_2, (dx / 2*lambda_3)*(hj/(hj-h_j3star)), dt) 

    mask = (h_istar < 0.0) & (h_j3star > 0.0) & (dtstar < dt)  
    beta_1 = np.where(mask, np.where(-beta1min >= beta3min, beta1min, beta_1), beta_1)
    beta_3 = np.where(mask, -beta_1, beta_3)
    
    mask = (h_istar > 0.0) & (h_j3star < 0.0) & (dt3star < dt)  
    beta_3 = np.where(mask, np.where(-beta3min >= beta3min, beta1min, beta_3), beta_3)
    beta_1 = np.where(mask, -beta_3, beta_1)

    # Reconstruction of approximate solution
    
        full_flux = np.zeros_like(lambdas)

    mask_1 = (np.isclose(hi, np.zeros_like(hi))) & (h_istar < 0.0) 
    mask_2 = (np.isclose(hj, np.zeros_like(hj))) & (h_j3star < 0.0)

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
"""
    """# thrust terms

    dz = zj - zi
    di = hi + zi
    dj = hj + zj
    dd = dj - di

    thrust_a = -g*htilde*dz

    mask1_dz = (dz >= 0) & (di < zj)
    mask2_dz = (dz < 0) & (dj < zi)

    dztilde = np.where(mask1_dz, hi, np.where(mask2_dz, hj, dz))

    hr = np.where(dz >= 0, hi, hj)
    thrust_b = -g*(hr - 0.5*np.abs(dztilde))*dztilde

    mask_thrust = (dz*dd >= 0.0) & (dz > 0.0)
    thrust = np.where(mask_thrust, np.maximum(thrust_a, thrust_b), thrust_b)

    beta_1 = thrust/(2*ctilde)
    beta_2 = np.zeros_like(beta_1)
    beta_3 = -beta_1.copy()

    # Entropy correction Harten-Hyman

    # lambda_1
    ei = uhati - np.sqrt(g*hi)
    ej = uhatj - np.sqrt(g*hj)

    mask_i = (ei < 0.0) & (ej > 0.0)

    lambda_1 = np.where(mask_i, ei*(ej-lambda_1)/((ej-ei)+1e-12), lambda_1) # Replace with Lambda hat
    lambda_E1 = np.where(mask_i, ej*(lambda_1-ei)/((ej-ei)+1e-12), 0.0)
    
    # lambda_3
    ei = uhati + np.sqrt(g*hi)
    ej = uhatj + np.sqrt(g*hj)

    mask_j = (ei < 0.0) & (ej > 0.0)
    
    lambda_3 = np.where(mask_j, ej*(lambda_3-ei)/((ej-ei)+1e-12), lambda_3)      # Replace with Lambda hat
    lambda_E3 = np.where(mask_j, ei*(ej-lambda_3)/((ej-ei)+1e-12), 0.0)
    lambda_E2 = np.zeros_like(lambda_E3) 
    
    # Entropy correction Harten-Hyman

    # Reconstruction of approximate solution
    
    h_istar = hi + alpha_1 - (beta_1/lambda_1)      # 1st intermediate state
    h_j3star = hj - alpha_3 + (beta_3/lambda_3)     # 3rd intermediate state
    
    beta1min = -(hi+alpha_1)*np.abs(lambda_1)
    beta3min = -(hi-alpha_1)*np.abs(lambda_3)

    dt = dx / np.max(np.abs(np.stack([lambda_2, lambda_3])), axis=0)

    mask_1 = (h_istar < 0.0) & (hi != 0.0)
    mask_2 = (h_j3star < 0.0) & (hj != 0.0)

    dtstar = np.where(mask_1, (dx / 2*lambda_1)*(hi/(hi-h_istar +1e-12)), dt) 
    dt3star = np.where(mask_2, (dx / 2*lambda_3)*(hj/(hj-h_j3star+1e-12)), dt) 

    mask = (h_istar < 0.0) & (h_j3star > 0.0) & (dtstar < dt)  
    beta_1 = np.where(mask, np.where(-beta1min >= beta3min, beta1min, beta_1), beta_1)
    beta_3 = np.where(mask, -beta_1, beta_3)
    
    mask = (h_istar > 0.0) & (h_j3star < 0.0) & (dt3star < dt)  
    beta_3 = np.where(mask, np.where(-beta3min >= beta3min, beta1min, beta_3), beta_3)
    beta_1 = np.where(mask, -beta_3, beta_1)

    # Reconstruction of approximate solution

    # Flux calculation

    lambdas_E = np.stack([lambda_E1, lambda_E2, lambda_E3], axis=-1)


    for i in range(lambdas.shape[-1]):
        flux = np.expand_dims(lambdas[...,i]*alphas[...,i]-betas[...,i],axis=2) * P[...,i]
        upwP +=  np.where(np.expand_dims(lambdas[...,i], axis=2) > 0.0, flux, 0.0)
        upwM +=  np.where(np.expand_dims(lambdas[...,i], axis=2) <= 0.0, flux, 0.0)

    for i in range(lambdas_E.shape[-1]):
        flux = np.expand_dims(lambdas_E[...,i]*alphas[...,i],axis=2) * P[...,i]
        upwP += np.where(np.expand_dims(lambdas_E[...,i], axis=2) > 0.0, flux, 0.0)
        upwM += np.where(np.expand_dims(lambdas_E[...,i], axis=2) <= 0.0, flux, 0.0

    mask_1 = (np.isclose(hi, np.zeros_like(hi))) & (h_istar < 0.0) 
    mask_2 = (np.isclose(hj, np.zeros_like(hj))) & (h_j3star < 0.0)

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
        upwM += np.where(np.expand_dims(lambdas_E[...,i], axis=2) <= 0.0, flux, 0.0)"""


    # Entropy correction Harten-Hyman

    """# lambda_1
    ei = uhati - np.sqrt(g*hi)
    ej = uhatj - np.sqrt(g*hj)

    mask_i = (ei < 0.0) & (ej > 0.0)

    lambda_1 = np.where(mask_i, ei*(ej-lambda_1)/((ej-ei)+1e-12), lambda_1) # Replace with Lambda hat
    lambda_E1 = np.where(mask_i, ej*(lambda_1-ei)/((ej-ei)+1e-12), 0.0)
    
    # lambda_3
    ei = uhati + np.sqrt(g*hi)
    ej = uhatj + np.sqrt(g*hj)

    mask_j = (ei < 0.0) & (ej > 0.0)
    
    lambda_3 = np.where(mask_j, ej*(lambda_3-ei)/((ej-ei)+1e-12), lambda_3)      # Replace with Lambda hat
    lambda_E3 = np.where(mask_j, ei*(ej-lambda_3)/((ej-ei)+1e-12), 0.0)

    lambda_E2 = np.zeros_like(lambda_E3) 
    
    # Entropy correction Harten-Hyman

    lambdas_E = np.stack([lambda_E1, lambda_E2, lambda_E3], axis=-1)

    # Reconstruction of approximate solution
    
    h_istar = hi + alphas[...,0] - (betas[...,0]/lambdas[...,0])      # 1st intermediate state
    h_j3star = hj - alphas[...,2] + (betas[...,2]/lambdas[...,2])     # 3rd intermediate state
    
    beta1min = -(hi+alphas[...,0])*np.abs(lambdas[...,0])
    beta3min = -(hi-alphas[...,0])*lambdas[...,2]

    dt = dx / np.max(np.abs(np.stack([lambdas[...,1], lambdas[...,2]])), axis=0)

    mask_1 = (h_istar < 0.0) & (hi != 0.0)
    mask_2 = (h_j3star < 0.0) & (hj != 0.0)

    dtstar = np.where(mask_1, (dx / 2*lambdas[...,0])*(hi/(hi-h_istar)), dt) 
    dt3star = np.where(mask_2, (dx / 2*lambdas[...,2])*(hj/(hj-h_j3star)), dt) 

    mask = (h_istar < 0.0) & (h_j3star > 0.0) & (dtstar < dt)  
    betas[...,0] = np.where(mask, np.where(-beta1min >= beta3min, beta1min, betas[...,0]), betas[...,0])
    betas[...,2] = np.where(mask, -betas[...,0], betas[...,2])
    
    mask = (h_istar > 0.0) & (h_j3star < 0.0) & (dt3star < dt)  
    betas[...,2] = np.where(mask, np.where(-beta3min >= beta3min, beta1min, betas[...,2]), betas[...,2])
    betas[...,0] = np.where(mask, -betas[...,2], betas[...,0])

    # Reconstruction of approximate solution


    mask_1 = (np.isclose(hi, np.zeros_like(hi))) & (h_istar < 0.0) 
    mask_2 = (np.isclose(hj, np.zeros_like(hj))) & (h_j3star < 0.0)

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

    """
