import numpy as np

def add_gausian_pulse(solver, laser):

    k0 = 2 * np.pi * laser['k0']
    a0 = laser['a0']

    Lx = laser['Lx']
    R = laser['R']
    x0 = laser['x0']
    x_foc = laser['x_foc']
    X_focus = x0 - x_foc

    Xgrid, Rgrid = solver.Args['Xgrid'],solver.Args['Rgrid'] # sin phase
    kx, w = solver.Args['kx'][None,:], solver.Args['w_m0']

    solver.DataDev['Ez_m0'][:] = a0 * np.sin(k0*(Xgrid[None,:]-x0)) \
          * np.exp(-(Xgrid[None,:]-x0)**2/Lx**2 - Rgrid[:,None]**2/R**2) \
          * (abs(Rgrid[:,None]) < 3.5*R) * (abs(Xgrid[None,:]-x0) < 3.5*Lx)

    solver.fb_transform(scals=['Ez',], dir=0)
    EE = solver.DataDev['Ez_fb_m0'].get()

    DT = -1.j * w * np.sign(kx + (kx==0))
    GG  = DT * EE

    EE_tmp = np.cos(w*X_focus) * EE + np.sin(w*X_focus) / w * GG
    GG = -w*np.sin(w*X_focus) * EE + np.cos(w*X_focus) * GG
    EE = EE_tmp
    EE *= np.exp(1.j * kx * X_focus)
    GG *= np.exp(1.j * kx * X_focus)

    solver.DataDev['Ez_fb_m0'][:] = EE
    solver.DataDev['Gz_fb_m0'][:] = GG

    solver.restore_B_fb()
    solver.fb_transform(vects=['B', 'E'], dir=1)
