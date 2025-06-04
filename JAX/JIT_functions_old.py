import jax 
import jax.numpy as jnp

g = 9.81

def _get_theta(_lambda, utilde, ctilde):
    return 3*_lambda**2 - 4*utilde*_lambda + utilde**2 - ctilde**2

def _get_approx_lambda_jit(_lambda, atilde, utilde, ctilde):
    theta = _get_theta(_lambda, utilde, ctilde)
    #denom = jnp.where(jnp.abs(theta - ctilde**2 * atilde) < 1e-10, 1e-10, theta - ctilde**2 * atilde)
    return (_lambda * theta - ctilde**2 * atilde * utilde) / (theta - ctilde**2*atilde)

def roe_solve_jit(si, sj,  nx, ny, dx):
    # define cells at both sides of the edge

    hi, hui, hvi, zi, gi = si
    hj, huj, hvj, zj, gj = sj

    ui = jnp.where(hi > 0.0, hui/hi, 0.0)
    vi = jnp.where(hi > 0.0, hvi/hi, 0.0)

    uhati = ui*nx + vi*ny   # normal direction
    vhati = -ui*ny + vi*nx  # tangential direction                              

    uj = jnp.where(hj > 0.0, huj/hj, 0.0)
    vj = jnp.where(hj > 0.0, hvj/hj, 0.0)
    
    uhatj = uj*nx + vj*ny   # normal direction
    vhatj = -uj*ny + vj*nx  # tangential direction      

    # get edge values
    sqrthi = jnp.sqrt(hi)
    sqrthj = jnp.sqrt(hj)

    # interface values
    htilde = 0.5 * (hi + hj)
    utilde = (uhati*sqrthi + uhatj*sqrthj) / (sqrthi+sqrthj)
    vtilde = (vhati*sqrthi + vhatj*sqrthj) / (sqrthi+sqrthj)
    ctilde = jnp.sqrt(g*htilde)
    gtilde = 0.5 * (gi + gj)
    atilde = gtilde*(uhati**2 + uhati*uhatj + uhatj**2 + vhati*vhatj)/jnp.sqrt(hi*hj) 
    
    lambda_1 = utilde - ctilde
    lambda_2 = utilde
    lambda_3 = utilde + ctilde

    Ptilde = jnp.stack([
        jnp.stack([jnp.ones_like(vtilde), jnp.zeros_like(vtilde), jnp.ones_like(vtilde)], axis=-1),  # row 1
        jnp.stack([utilde-ctilde,         jnp.zeros_like(vtilde), utilde+ctilde],         axis=-1),  # row 2
        jnp.stack([vtilde,                jnp.ones_like(vtilde),  vtilde],                axis=-1),   # row 3
    ], axis=-2)

    # projection calculation
        
    dh = hj - hi
    dhu = hj*uhatj - hi*uhati
    dhv = hj*vhatj - hi*vhati
    dz = zj - zi

    alpha_1 = ((utilde+ctilde)*dh - dhu)/(2*ctilde) 
    alpha_2 = -vtilde*dh + dhv 
    alpha_3 = (-(utilde-ctilde)*dh + dhu)/(2*ctilde)

    beta_1 = 0.5*(g*htilde*dz)/ctilde
    beta_2 = jnp.zeros_like(beta_1)
    beta_3 = -0.5*(g*htilde*dz)/ctilde

    dt = jnp.min(dx / jnp.max(jnp.stack([lambda_2, lambda_3]), axis=0))
    
    # Entropy correction
    # Harten-Hyman entropy fix
    # doi:10.1016/j.jcp.2010.02.016 eq (153, 154)

    lambda_11 = uhati - jnp.sqrt(g*hi)
    lambda_12 = uhatj - jnp.sqrt(g*hj)

    mask_E1 = (lambda_11 < 0.0) & (lambda_12 > 0.0)

    lambda_flat1 = lambda_11 * (lambda_12 - lambda_1)/(lambda_12 - lambda_11)

    lambda_E1 = jnp.where(mask_E1, lambda_1 - lambda_flat1, 0.0)   # lambda hat  - only works on U projecttion
    lambda_1 = jnp.where(mask_E1, lambda_flat1, lambda_1)           # lambda flat - works on projection and sources

    lambda_31 = uhati + jnp.sqrt(g*hi)
    lambda_32 = uhatj + jnp.sqrt(g*hj)

    mask_E3 = (lambda_31 < 0.0) & (lambda_32 > 0.0)

    lambda_flat3 = lambda_32 * (lambda_3 - lambda_31)/(lambda_32 - lambda_31)

    lambda_E3 = jnp.where(mask_E3, lambda_3 - lambda_flat3, 0.0)   # lambda hat  - only works on U projecttion
    lambda_3 = jnp.where(mask_E3, lambda_flat3, lambda_3)           # lambda flat - works on projection and sources

    lambda_E2 = jnp.zeros_like(lambda_2)
    # Entropy correction

    # wet-wet subcritical correction 
    hi_star1 = hi + alpha_1 - beta_1/lambda_1
    hj_star3 = hj - alpha_3 + beta_3/lambda_3

    dtstar1 = dh*hi / ((2*lambda_1)*(hi-hi_star1))
    dtstar3 = dh*hj / ((2*lambda_2)*(hj-hj_star3))
    
    beta1_min = -(hi + alpha_1)*jnp.abs(lambda_1)
    beta3_min = -(hj - alpha_3)*lambda_3

    mask1 = (hi_star1 < 0.0) & (hj_star3 > 0.0) & (dtstar1 < dt) & (-beta1_min > beta3_min)
    mask2 = (hi_star1 > 0.0) & (hj_star3 < 0.0) & (dtstar3 < dt) & (-beta3_min > beta1_min)
    
    beta_1 = jnp.where(mask1, beta1_min, beta_1)
    beta_3 = jnp.where(mask1, -beta_1, beta_3)

    beta_3 = jnp.where(mask2, beta3_min, beta_3)
    beta_1 = jnp.where(mask2, -beta_3, beta_1)
    lambdas_E = jnp.stack([lambda_E1, lambda_E2, lambda_E3], axis=-1)
    # wet-wet subcritical correction

    lambdas = jnp.stack([lambda_1, lambda_2, lambda_3], axis=-1)
    alphas = jnp.stack([alpha_1, alpha_2, alpha_3], axis=-1)
    betas = jnp.stack([beta_1, beta_2, beta_3], axis=-1)

    upwP = jnp.zeros((htilde.shape[0], htilde.shape[1], 3))
    upwM = jnp.zeros((htilde.shape[0], htilde.shape[1], 3))

    for i in range(lambdas.shape[2]):
        contribution = jnp.expand_dims(lambdas[..., i]*alphas[..., i]-betas[..., i], axis=2)*Ptilde[..., i]*dx
        upwP = upwP.at[:].add(jnp.where(jnp.expand_dims(lambdas[..., i],axis=2) > 0.0, contribution, 0.0))
        upwM = upwM.at[:].add(jnp.where(jnp.expand_dims(lambdas[..., i],axis=2) <= 0.0, contribution, 0.0)) 
    
    
    for i in range(lambdas_E.shape[2]):
        contribution = jnp.expand_dims(jnp.expand_dims(lambdas_E[..., i]*alphas[..., i], axis=2)*Ptilde[..., i])
        upwP = upwP.at[:].add(jnp.where(jnp.expand_dims(lambdas_E[..., i],axis=2) > 0.0, contribution, 0.0))
        upwM = upwM.at[:].add(jnp.where(jnp.expand_dims(lambdas_E[..., i],axis=2) <= 0.0, contribution, 0.0))
    

    return upwP, upwM, dt


def roe_solve_2D_jit(fluxes, h, hu, hv, z, G, dx):
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

    upwP_x, upwM_x, dt_x = roe_solve_jit(s1_x, s2_x, 1, 0, dx)
    upwP_y, upwM_y, dt_y = roe_solve_jit(s1_y, s2_y, 0, -1, dx)

    # upwinding solution

    fluxes = fluxes.at[1:-1,:-1].add(upwM_x) 
    fluxes = fluxes.at[1:-1, 1:].add(upwP_x)
    fluxes = fluxes.at[:-1,1:-1].add(upwM_y) 
    fluxes = fluxes.at[1: ,1:-1].add(upwP_y)
    
    dt = jnp.minimum(dt_x, dt_y)

    return fluxes, dt


def exner_solve_jit(s1, s2,  nx, ny, dx):
    # We will be implementing the Approximately Coupled Solver (ACM) https://doi.org/10.1016/j.advwatres.2021.103931
    hi, hui, hvi, zi, gi = s1
    hj, huj, hvj, zj, gj = s2    

    # get bedload wave celerity

    ui = jnp.where(hi > 0.0, hui/hi, 0.0)
    vi = jnp.where(hi > 0.0, hvi/hi, 0.0)

    uhati = ui*nx + vi*ny   # normal direction
    vhati = -ui*ny + vi*nx  # tangential direction                              

    uj = jnp.where(hj > 0.0, huj/hj, 0.0)
    vj = jnp.where(hj > 0.0, hvj/hj, 0.0)
    
    uhatj = uj*nx + vj*ny   # normal direction
    vhatj = -uj*ny + vj*nx  # tangential direction

    qb_nhati = gi*(uhati**2 + vhati**2)*uhati
    qb_nhatj = gi*(uhatj**2 + vhatj**2)*uhatj
    
    gtilde = 0.5*(gi+gj)
    dz = zj - zi
    dz_safe = jnp.where(jnp.abs(dz) < 1e-8, 1e-8, dz)

    # figure this out later
    #qbhatL = gtilde*(uhati**2+vhati**2)*uhati - gi*(uhati**2+vhati**2)*uhati 
    dqbhat = gtilde*(uhatj**2+vhatj**2)*uhatj - gtilde*(uhati**2+vhati**2)*uhati 
    #qbhatR = gj*(uhatj**2+vhatj**2)*uhatj - gtilde*(uhatj**2+vhatj**2)*uhatj
    #dqbhat = qbhatL + qbhat + qbhatR

    #dqbhat = qb_nhati - qb_nhatj
    lambda_4 = dqbhat / dz_safe

    corrector_i = (gtilde - gi)*(uhati**2 + vhati**2)*uhati
    corrector_j = (gtilde - gj)*(uhatj**2 + vhatj**2)*uhatj

    upwP = (jnp.where(lambda_4 > 0.0, qb_nhati, qb_nhatj)+corrector_i)*dx 
    upwM = (jnp.where(lambda_4 <= 0.0, qb_nhatj, qb_nhati)+corrector_j)*dx 

    return upwP, upwM

def exner_solve_2D_jit(fluxes, h, hu, hv, z, G, dh):
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

    upwP_x, upwM_x = exner_solve_jit(s1_x, s2_x, 1, 0, dh)
    upwP_y, upwM_y = exner_solve_jit(s1_y, s2_y, 0, -1, dh)

    fluxes = fluxes.at[1:-1,:-1].add(upwM_x) 
    fluxes = fluxes.at[1:-1, 1:].add(upwP_x)
    fluxes = fluxes.at[:-1,1:-1].add(upwM_y)
    fluxes = fluxes.at[1: ,1:-1].add(upwP_y)

    return fluxes