import numpy as np

# Constants
g = 9.81 # Gravity

def _get_theta(_lambda, utilde, ctilde):
    return 3*_lambda**2 - 4*utilde*_lambda + utilde**2 - ctilde**2

def _get_approx_lambda(_lambda, atilde, utilde, ctilde):
    theta = _get_theta(_lambda, utilde, ctilde)
    return (_lambda * theta - ctilde**2 * atilde * utilde) / (theta - ctilde**2*atilde)

def compute_dt_SWE(h,hu,hv,dx):
    dt_x = np.min(dx / (np.abs(hu[1:-1,1:-1]/h[1:-1,1:-1])+np.sqrt(g*h[1:-1,1:-1])))
    dt_y = np.min(dx / (np.abs(hv[1:-1,1:-1]/h[1:-1,1:-1])+np.sqrt(g*h[1:-1,1:-1])))
    
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
    atilde = gtilde*(uhati**2 + uhati*uhatj + uhatj**2 + vhati*vhatj)/np.sqrt(hi*hj) 
    
    # calculate wave speeds 

    lambda_1 = utilde - ctilde
    lambda_2 = utilde
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

    hi, hui, hvi, zi, gi, ni = si
    hj, huj, hvj, zj, gj, nj = sj

    sqrt_i = np.sqrt(hi)
    sqrt_j = np.sqrt(hj)

    ui = np.where(hi > 0.0, hui/hi, 0.0)
    vi = np.where(hi > 0.0, hvi/hi, 0.0)

    uj = np.where(hj > 0.0, huj/hj, 0.0)
    vj = np.where(hj > 0.0, hvj/hj, 0.0)

    # rotate problem so that it is in the (n,t) coordinate frame

    uhati = ui*nx + vi*ny          # normal 
    vhati = -ui*ny + vi*nx         # tangent
    
    uhatj = uj*nx + vj*ny            # normal
    vhatj = -uj*ny + vj*nx           # tangent

    htilde = 0.5*(hi + hj)
    utilde = (uhati*sqrt_i + uhatj*sqrt_j)/(sqrt_i + sqrt_j)
    vtilde = (vhati*sqrt_i + vhatj*sqrt_j)/(sqrt_i + sqrt_j)
    ctilde = np.sqrt(g*htilde)

    # calculate wave speeds 

    lambda_1 = utilde - ctilde
    lambda_2 = utilde
    lambda_3 = utilde + ctilde

    P = np.stack([
        np.stack([np.ones_like(vtilde), np.zeros_like(vtilde), np.ones_like(vtilde)],axis=-1),
        np.stack([lambda_1,             np.zeros_like(vtilde),             lambda_3],axis=-1),
        np.stack([vtilde,               ctilde,                              vtilde],axis=-1)
    ], axis=-2)

    dh = hj - hi
    dhu = hj*uhatj - hi*uhati
    dhv = hj*vhatj - hi*vhati
    dz = zj - zi

    alpha_1 = ((ctilde-utilde)*dh - dhu)/(2*ctilde)
    alpha_2 = (-vtilde*dh + dhv)/ctilde
    alpha_3 = ((utilde+ctilde)*dh + dhu)/(2*ctilde)

    coef = g*hi*(ni**2*hi**(-4/3))*np.sqrt(uhati**2*vhati**2)
    t_in = - coef*uhati

    coef = g*hj*(nj**2*hj**(-4/3))*np.sqrt(uhatj**2*vhatj**2)
    t_jn = - coef*uhatj

    t_tilde = 0.5*(t_jn + t_in)*dx
    s_tilde = -g*htilde*dz

    beta_1 = -0.5*(s_tilde + t_tilde)/ctilde
    beta_2 = np.zeros_like(beta_1)
    beta_3 = 0.5*(s_tilde + t_tilde)/ctilde

    # Flux calculation

    lambdas = np.stack([lambda_1, lambda_2, lambda_3], axis=-1)
    alphas = np.stack([alpha_1, alpha_2, alpha_3], axis=-1)
    betas = np.stack([beta_1, beta_2, beta_3], axis=-1)
    
    upwP = np.zeros_like(lambdas)
    upwM = np.zeros_like(lambdas)

    for i in range(lambdas.shape[-1]):
        flux = np.expand_dims(lambdas[...,i]*alphas[...,i]-betas[...,i],axis=2) * P[...,i]
        upwP +=  np.where(np.expand_dims(lambdas[...,i], axis=2) > 0.0, flux, 0.0)
        upwM +=  np.where(np.expand_dims(lambdas[...,i], axis=2) < 0.0, flux, 0.0)


    Tk_inv = np.array([[1,  0,   0],
                       [0, nx, -ny],
                       [0, ny,  nx]])

    upwP = np.einsum('ij,klj->kli', Tk_inv, upwP) 
    upwM = np.einsum('ij,klj->kli', Tk_inv, upwM)

    return upwP, upwM


def roe_solve_2D(fluxes, h, hu, hv, z, G, n, dx):
    hi_x =   h[:,:-1]
    hui_x = hu[:,:-1]
    hvi_x = hv[:,:-1]
    zi_x =   z[:,:-1]
    gi_x =   G[:,:-1]
    ni_x =   n[:,:-1]

    hj_x =   h[:, 1:]
    huj_x = hu[:, 1:]
    hvj_x = hv[:, 1:]
    zj_x =   z[:, 1:]
    gj_x =   G[:, 1:]
    nj_x =   n[:, 1:]

    hi_y =   h[:-1,:]
    hui_y = hu[:-1,:]
    hvi_y = hv[:-1,:]
    zi_y =   z[:-1,:]
    gi_y =   G[:-1,:]
    ni_y =   n[:-1,:]

    hj_y =   h[1:, :]
    huj_y = hu[1:, :]
    hvj_y = hv[1:, :]
    zj_y =   z[1:, :]
    gj_y =   G[1:, :]
    nj_y =   n[1:, :]
    
    s1_x = np.stack([
        hi_x,hui_x,hvi_x,zi_x,gi_x,ni_x
    ])

    s2_x = np.stack([
        hj_x,huj_x,hvj_x,zj_x,gj_x,nj_x
    ])
    
    s1_y = np.stack([
        hi_y,hui_y,hvi_y,zi_y,gi_y,ni_y
    ])

    s2_y = np.stack([
        hj_y,huj_y,hvj_y,zj_y,gj_y,nj_y
    ])

    upwP_x, upwM_x = roe_solver(s1_x, s2_x, 1,  0, dx)
    upwP_y, upwM_y = roe_solver(s1_y, s2_y, 0, -1, dx)

    # upwinding solution

    fluxes[:,:-1] += upwM_x 
    fluxes[:-1,:] += upwM_y 
    fluxes[:, 1:] += upwP_x 
    fluxes[1:, :] += upwP_y
    
    return fluxes
    
    
def exner_solve(s1, s2,  nx, ny, dx):
    # We will be implementing the Approximately Coupled Solver (ACM) https://doi.org/10.1016/j.advwatres.2021.103931
    hi, hui, hvi, zi, gi, ni = s1
    hj, huj, hvj, zj, gj, nj = s2

    sqrt_i = np.sqrt(hi)
    sqrt_j = np.sqrt(hj)

    ui = np.where(hi > 0.0, hui/hi, 0.0)
    vi = np.where(hi > 0.0, hvi/hi, 0.0)

    uhati = ui*nx + vi*ny   # normal direction
    vhati = -ui*ny + vi*nx  # tangential direction

    uj = np.where(hj > 0.0, huj/hj, 0.0)
    vj = np.where(hj > 0.0, hvj/hj, 0.0)
    
    uhatj = uj*nx + vj*ny   # normal direction
    vhatj = -uj*ny + vj*nx  # tangential direction

    htilde = 0.5*(hi + hj)
    utilde = (uhati*sqrt_i + uhatj*sqrt_j)/(sqrt_i + sqrt_j)
    vtilde = (vhati*sqrt_i + vhatj*sqrt_j)/(sqrt_i + sqrt_j)
    ctilde = np.sqrt(g*htilde)

    qb_nhati = gi*(uhati**2 + vhati**2)*uhati
    qb_nhatj = gj*(uhatj**2 + vhatj**2)*uhatj

    gtilde = 0.5*(gi+gj)
    dz = zj - zi + 1e-12

    # figure this out later
    #qbhatL = gtilde*(uhati**2+vhati**2)*uhati - gi*(uhati**2+vhati**2)*uhati 
    dqbhat = gtilde*(uhatj**2+vhatj**2)*uhatj - gtilde*(uhati**2+vhati**2)*uhati 
    #qbhatR = gj*(uhatj**2+vhatj**2)*uhatj - gtilde*(uhatj**2+vhatj**2)*uhatj
    #dqbhat = qbhatL + qbhat + qbhatR
    #dqbhat = qb_nhatj - qb_nhati

    lambda_4 = np.where(np.abs(dz) > 1e-9, dqbhat / dz, utilde)

    corrector_i = (gtilde - gi)*(uhati**2 + vhati**2)*uhati
    corrector_j = (gtilde - gj)*(uhatj**2 + vhatj**2)*uhatj
    
    upwP = np.where(lambda_4 >= 0.0,
                    qb_nhati + corrector_i,  # use left state
                    qb_nhatj + corrector_j)  # use right state

    upwM = np.where(lambda_4 <  0.0,
                    qb_nhatj + corrector_j,  # use right state
                    qb_nhati + corrector_i)  # use left state

    
    return upwP, upwM

def exner_solve_2D(fluxes, h, hu, hv, z, qb_x, qb_y, G, n, dx):
    hi_x =   h[:,:-1]
    hui_x = hu[:,:-1]
    hvi_x = hv[:,:-1]
    zi_x =   z[:,:-1]
    gi_x =   G[:,:-1]
    ni_x =   n[:,:-1]

    hj_x =   h[:, 1:]
    huj_x = hu[:, 1:]
    hvj_x = hv[:, 1:]
    zj_x =   z[:, 1:]
    gj_x =   G[:, 1:]
    nj_x =   n[:, 1:]

    hi_y =   h[:-1,:]
    hui_y = hu[:-1,:]
    hvi_y = hv[:-1,:]
    zi_y =   z[:-1,:]
    gi_y =   G[:-1,:]
    ni_y =   n[:-1,:]

    hj_y =   h[1:, :]
    huj_y = hu[1:, :]
    hvj_y = hv[1:, :]
    zj_y =   z[1:, :]
    gj_y =   G[1:, :]
    nj_y =   n[1:, :]
    
    s1_x = np.stack([
        hi_x,hui_x,hvi_x,zi_x,gi_x,ni_x
    ])

    s2_x = np.stack([
        hj_x,huj_x,hvj_x,zj_x,gj_x,nj_x
    ])
    
    s1_y = np.stack([
        hi_y,hui_y,hvi_y,zi_y,gi_y,ni_y
    ])

    s2_y = np.stack([
        hj_y,huj_y,hvj_y,zj_y,gj_y,nj_y
    ])

    upwP_x, upwM_x = exner_solve(s1_x, s2_x, 1, 0, dx)
    upwP_y, upwM_y = exner_solve(s1_y, s2_y, 0, -1, dx)

    fluxes[:,:-1] += upwM_x 
    fluxes[:-1,:] += upwM_y 
    fluxes[:, 1:] -= upwP_x 
    fluxes[1:, :] -= upwP_y

    qb_x[:,:-1] = upwM_x 
    qb_x[:, 1:] += upwP_x 
    qb_y[:-1,:] = -upwM_y 
    qb_y[1:, :] -= upwP_y    

    return fluxes, qb_x, qb_y