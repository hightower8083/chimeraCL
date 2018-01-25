// this is a source of particles kernels for chimeraCL project
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64:enable

// Multiply particles weight by an intrpolant profile
__kernel void profile_by_interpolant(
  __global double *x,
  __global double *w,
           uint Np,
  __global double *xx_loc,
  __global double *ff_loc,
  __global double *dxm1_loc,
           uint Nx_loc)
{
  uint ip = (uint) get_global_id(0);
  if (ip < Np)
   {
     double xp = x[ip];
     uint ix;

     for (ix=0; ix<Nx_loc-1; ix++){
       if (xp>xx_loc[ix] && xp<=xx_loc[ix+1]) {break;}
     }

     double f_minus = ff_loc[ix]*dxm1_loc[ix];
     double f_plus = ff_loc[ix+1]*dxm1_loc[ix];

     w[ip] *= f_minus*(xx_loc[ix+1]-x[ip]) + f_plus*(x[ip]-xx_loc[ix]);
   }
}

// Fill given grid with uniformly distributed particles
__kernel void fill_grid(
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *w,
  __global double *theta_var,
  __global double *xgrid,
  __global double *rgrid,
           uint Nx,
           uint ncells,
           uint Nppc_x,
           uint Nppc_r,
           uint Nppc_th)
{
    uint i_cell = (uint) get_global_id(0);
    if (i_cell < ncells)
    {
        uint Nx_cell = Nx-1;
        uint Nppc_loc = Nppc_x*Nppc_r*Nppc_th;

        uint ir =  i_cell/Nx_cell;
        uint ix =  i_cell - Nx_cell*ir;
        uint ip = i_cell*Nppc_loc;

        double xmin = xgrid[ix];
        double rmin = rgrid[ir];
        double thmin = theta_var[i_cell];

        double Lx = xgrid[ix+1] - xgrid[ix];
        double Lr = rgrid[ir+1] - rgrid[ir];
        double dx = 1./( (double) Nppc_x);
        double dr = 1./( (double) Nppc_r);
        double dth = 2*M_PI/( (double) Nppc_th);
        double th, rp, sin_th, cos_th, rp_s, rp_c;

        for (uint incell_th=0; incell_th<Nppc_th; incell_th++){
          th = thmin + incell_th*dth;
          sin_th = sin(th);
          cos_th = cos(th);
          for (int incell_r=0;incell_r<Nppc_r;incell_r++){
            rp = rmin + (0.5+incell_r)*dr*Lr;
            rp_s = rp*sin_th;
            rp_c = rp*cos_th;
            for (int incell_x=0;incell_x<Nppc_x;incell_x++){
              x[ip] = xmin + (0.5+incell_x)*dx*Lx;
              y[ip] = rp_s;
              z[ip] = rp_c;
              w[ip] = rp;
              ip += 1;
            }}}
  }
}

// Find cell indicies of the particles and
// sums of paricles per cell
__kernel void index_and_sum_in_cell(
  __global double *x,
  __global double *y,
  __global double *z,
  __global uint *sum_in_cell,
  __constant uint *num_p,
  __global uint *indx_in_cell,
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv)
{
  uint ip = (uint) get_global_id(0);
  if (ip < *num_p)
   {
    double r;
    int ix,ir;
    int Nx_loc = (int) *Nx-1;
    int Nr_loc = (int) *Nr-1;

    r = sqrt(y[ip]*y[ip]+z[ip]*z[ip]);

    ix = (int)floor( (x[ip] - *xmin)*(*dx_inv) );
    ir = (int)floor((r - *rmin)*(*dr_inv));

    if (ix > 0 && ix < Nx_loc-1 && ir < Nr_loc-1 && ir >= 0)
     {
      indx_in_cell[ip] = ix + ir * Nx_loc;
      atomic_add(&sum_in_cell[indx_in_cell[ip]], 1U);
//      atom_add(&sum_in_cell[indx_in_cell[ip]], 1U);
     }
    else
     {
      indx_in_cell[ip] = Nr_loc*Nx_loc ;
      atomic_add(&sum_in_cell[indx_in_cell[ip]], 1U);
//      atom_add(&sum_in_cell[indx_in_cell[ip]], 1U);
     }
  }
}

// Advance particles coordinates
__kernel void push_xyz(
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *px,
  __global double *py,
  __global double *pz,
  __global double *g_inv,
  __constant double *dt,
  __constant uint *num_p)
{
  uint ip = (uint) get_global_id(0);
  if (ip < *num_p)
   {
    double dt_g = (*dt) * g_inv[ip];

    double dx = px[ip] * dt_g;
    double dy = py[ip] * dt_g;
    double dz = pz[ip] * dt_g;

    x[ip] = x[ip] + dx;
    y[ip] = y[ip] + dy;
    z[ip] = z[ip] + dz;
   }
}

// Copy sorted particle data of double-type to a new array
__kernel void data_align_dbl(
  __global double *x,
  __global double *x_new,
  __global uint *sorted_indx,
  uint num_p)
{
  uint ip = (uint) get_global_id(0);
  double x_tmp;
  if (ip < num_p)
  {
   x_tmp = x[sorted_indx[ip]];
   x_new[ip] =  x_tmp;
  }
}

// Copy sorted particle data of integer-type to a new array
__kernel void data_align_int(
  __global uint *x,
  __global uint *x_new,
  __global uint *sorted_indx,
  __constant uint *num_p)
{
  uint ip = (uint) get_global_id(0);
  if (ip < *num_p)
   {
    x_new[ip] = x[sorted_indx[ip]];
   }
}


__kernel void sort(
  __global uint *cell_offset,
  __global uint *indx_in_cell,
  __global uint *new_sum_in_cell,
  __global uint *sorted_indx,
                  uint num_p)
{
  uint ip = (uint) get_global_id(0);
  if (ip < num_p)
   {
    uint i_cell = indx_in_cell[ip];
    uint ip_offset_loc = atomic_add(&new_sum_in_cell[i_cell], 1U);
    uint ip_sorted = cell_offset[i_cell] + ip_offset_loc;
    sorted_indx[ip_sorted] = ip;
   }
}
