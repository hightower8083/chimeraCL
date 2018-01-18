// Linear projection of a weighted vector of particles onto 2D grid
__kernel void gather_and_push(
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *px,
  __global double *py,
  __global double *pz,
  __global double *g_inv,
  __global uint *sorting_indx,
  __global uint *indx_offset,
  __constant double *dt,
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv,
  __constant uint *Nxm1Nrm1,
  __global double *ex_m0,
  __global double *ey_m0,
  __global double *ez_m0,
  __global double *bx_m0,
  __global double *by_m0,
  __global double *bz_m0,
  __global double2 *ex_m1,
  __global double2 *ey_m1,
  __global double2 *ez_m1,
  __global double2 *bx_m1,
  __global double2 *by_m1,
  __global double2 *bz_m1)
{
  // running kernels over the 4cells
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *Nxm1Nrm1)
   {
    // get cell number and period of grid
    uint Nx_grid = *Nx;
    uint Nx_cell = Nx_grid - 1;

    // get indicies of cell
    uint ir = i_cell/Nx_cell;
    uint ix = i_cell - ir*Nx_cell;

    // get 1D indicies of the selected
    // cell and grid node on the global grid
    uint i_cell_glob = ix + ir*Nx_cell;
    uint i_grid_glob = ix + ir*Nx_grid;

    // get particles indicies in the selected cell
    uint ip_start = indx_offset[i_cell_glob];
    uint ip_end = indx_offset[i_cell_glob+1];

    // skip empty cells
    if (ip_start == ip_end) {return;}

    // allocate few integer counters
    uint i,j,k,i_dep;

    // allocate privite cell array for the deposition
    // NB: multiplictaion of m1 by 2 accounts for Hermit symmetry
    double e_cell_m0[3][2][2] ;
    double b_cell_m0[3][2][2] ;
    double e_cell_m1[3][2][2][2] ;
    double b_cell_m1[3][2][2][2] ;

    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        i_dep = i_grid_glob + j + Nx_grid*i;
        e_cell_m0[0][i][j] = ex_m0[i_dep];
        e_cell_m0[1][i][j] = ey_m0[i_dep];
        e_cell_m0[2][i][j] = ez_m0[i_dep];

        b_cell_m0[0][i][j] = bx_m0[i_dep];
        b_cell_m0[1][i][j] = by_m0[i_dep];
        b_cell_m0[2][i][j] = bz_m0[i_dep];

        e_cell_m1[0][i][j][0] = 2 * ((double) ex_m1[i_dep].s0);
        e_cell_m1[0][i][j][1] = 2 * ((double) ex_m1[i_dep].s1);

        e_cell_m1[1][i][j][0] = 2 * ((double) ey_m1[i_dep].s0);
        e_cell_m1[1][i][j][1] = 2 * ((double) ey_m1[i_dep].s1);

        e_cell_m1[2][i][j][0] = 2 * ((double) ez_m1[i_dep].s0);
        e_cell_m1[2][i][j][1] = 2 * ((double) ez_m1[i_dep].s1);

        b_cell_m1[0][i][j][0] = 2 * ((double) bx_m1[i_dep].s0);
        b_cell_m1[0][i][j][1] = 2 * ((double) bx_m1[i_dep].s1);

        b_cell_m1[1][i][j][0] = 2 * ((double) by_m1[i_dep].s0);
        b_cell_m1[1][i][j][1] = 2 * ((double) by_m1[i_dep].s1);

        b_cell_m1[2][i][j][0] = 2 * ((double) bz_m1[i_dep].s0);
        b_cell_m1[2][i][j][1] = 2 * ((double) bz_m1[i_dep].s1);
        }}

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double C_cell[2][2];
    double exp_m1[2];

    double e_p[3];
    double b_p[3];
    double u_p[3];

    double dt_2 = 0.5*(*dt);
    double um[3], up[3], u0[3], t[3], s[3], t2p1_m1_05, g_p_inv;

    double xp, yp, zp, rp, rp_inv;
    double rmin_loc = *rmin;
    double dr_inv_loc = *dr_inv;
    double xmin_loc = *xmin;
    double dx_inv_loc = *dx_inv;
    uint ip_srtd;

    // run over the particles for linear deposition
    for (uint ip=ip_start; ip<ip_end; ip++){
      ip_srtd = sorting_indx[ip];
      xp = x[ip_srtd];
      yp = y[ip_srtd];
      zp = z[ip_srtd];

      rp = sqrt(yp*yp + zp*zp);

      rp_inv = 1. / rp;
      exp_m1[0] = yp * rp_inv;
      exp_m1[1] = -zp * rp_inv;

      sX1 = (xp - xmin_loc)*dx_inv_loc - ix;
      sX0 = 1.0 - sX1;
      sR1 = (rp - rmin_loc)*dr_inv_loc - ir;
      sR0 = 1.0 - sR1;

      C_cell[0][0] = sR0 * sX0;
      C_cell[0][1] = sR0 * sX1;
      C_cell[1][0] = sR1 * sX0;
      C_cell[1][1] = sR1 * sX1;

      for (k=0;k<3;k++){
        e_p[k] = 0;
        b_p[k] = 0;
        }

      for (k=0;k<3;k++){
        for (i=0;i<2;i++){
          for (j=0;j<2;j++){
            e_p[k] += C_cell[i][j] * e_cell_m0[k][i][j];
            b_p[k] += C_cell[i][j] * b_cell_m0[k][i][j];

            e_p[k] += C_cell[i][j] * e_cell_m1[k][i][j][0] * exp_m1[0];
            e_p[k] -= C_cell[i][j] * e_cell_m1[k][i][j][1] * exp_m1[1];
            b_p[k] += C_cell[i][j] * b_cell_m1[k][i][j][0] * exp_m1[0];
            b_p[k] -= C_cell[i][j] * b_cell_m1[k][i][j][1] * exp_m1[1];
            }}}

      u_p[3] = {px[ip_srtd], py[ip_srtd], pz[ip_srtd]};

      for(k=0;k<3;k++){
        um[k] = u_p[k] + dt_2*e_p[k];
      }

      g_p_inv = 1. / sqrt(1. + um[0]*um[0] + um[1]*um[1] + um[2]*um[2]);

      for(k=0;k<3;k++){
        t[k] = dt_2 * b_p[k] * g_p_inv;
        }

      t2p1_m1_05 = 2. / (1. + t[0]*t[0] + t[1]*t[1] + t[2]*t[2]) ;

      for(k=0;k<3;k++){
        s[k] = t[k] * t2p1_m1_05;
        }

      u0[0] = um[0] + um[1]*t[2] - um[2]*t[1];
      u0[1] = um[1] - um[0]*t[2] + um[2]*t[0];
      u0[2] = um[2] + um[0]*t[1] - um[1]*t[0];

      up[0] = um[0] +  u0[1]*s[2] - u0[2]*s[1];
      up[1] = um[1] -  u0[0]*s[2] + u0[2]*s[0];
      up[2] = um[2] +  u0[0]*s[1] - u0[1]*s[0];

      for(int k=0;k<3;k++) {
        u_p[k] = up[k] + dt_2*b_p[k];
        }

      g_p_inv = 1. / sqrt(1. + u_p[0]*u_p[0] + u_p[1]*u_p[1] + u_p[2]*u_p[2]);

      px[ip_srtd] = u_p[0];
      py[ip_srtd] = u_p[1];
      pz[ip_srtd] = u_p[2];
      g_inv[ip_srtd] = g_p_inv;
      }
  }
}
