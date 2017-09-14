// this is a source of particles kernels for chimeraCL project
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// Copy sorted particle data of double-type to a new array
__kernel void data_align_dbl(
  __global double *x,
  __global double *x_new,
  __global uint *sorted_indx,
  uint num_p)
{
  uint ip = get_global_id(0);
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
  uint ip = get_global_id(0);
  if (ip < *num_p)
   {
    x_new[ip] = x[sorted_indx[ip]];
   }
}

// Find cell indicies of the particles and
// sums of paricles per cell
__kernel void index_and_sum_in_cell(
  __global double *x,
  __global double *y,
  __global double *z,
  __global uint *indx_in_cell,
  __global uint *sum_in_cell,
  __constant uint *num_p,
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv)
{
  uint ip = get_global_id(0);
  if (ip < *num_p)
   {
    double r;
    int ix,ir;
    uint Nx_loc = *Nx-1;
    uint Nr_loc = *Nr-1;

    r = sqrt(y[ip]*y[ip]+z[ip]*z[ip]);

    ix = (int)floor( (x[ip] - *xmin)*(*dx_inv) );
    ir = (int)floor((r - *rmin)*(*dr_inv));

    if (ix >= 0 && ix < Nx_loc && ir < Nr_loc && ir >= 0)
     {
      indx_in_cell[ip] = ix + ir * Nx_loc;
      atom_add(&sum_in_cell[indx_in_cell[ip]], 1U);
     }
    else
     {
      indx_in_cell[ip] = (Nr_loc+1) * (Nx_loc+1) + 1;
     }
  }
}

// Advance particles momenta using Boris pusher
// and write the inverse Lorentz factor array
__kernel void push_p_boris(
  __global double *px,
  __global double *py,
  __global double *pz,
  __global double *g_inv,
  __global double *Ex,
  __global double *Ey,
  __global double *Ez,
  __global double *Bx,
  __global double *By,
  __global double *Bz,
  __constant double *dt,
  __constant uint *num_p)
{
  uint ip = get_global_id(0);
  if (ip < *num_p)
   {
    double u_p[3] = {px[ip],py[ip],pz[ip]};
    double E_p[3] = {Ex[ip],Ey[ip],Ez[ip]};
    double B_p[3] = {Bx[ip],By[ip],Bz[ip]};

    double dt_2 = 0.5*(*dt);
    double um[3], up[3], u0[3], t[3], t2, s[3], g_p;

    for(int i=0;i<3;i++){
      um[i] = u_p[i] + dt_2*E_p[i];
      }

    g_p = sqrt( 1. + um[0]*um[0] + um[1]*um[1] + um[2]*um[2]);

    for(int i=0;i<3;i++){
      t[i] = dt_2*B_p[i]/g_p;
      }

    t2 = t[0]*t[0] + t[1]*t[1] + t[2]*t[2];

    for(int i=0;i<3;i++){
      s[i] = 2*t[i]/(1+t2);
      }

    u0[0] = um[0] + um[1]*t[2] - um[2]*t[1];
    u0[1] = um[1] - um[0]*t[2] + um[2]*t[0];
    u0[2] = um[2] + um[0]*t[1] - um[1]*t[0];

    up[0] = um[0] +  u0[1]*s[2] - u0[2]*s[1];
    up[1] = um[1] -  u0[0]*s[2] + u0[2]*s[0];
    up[2] = um[2] +  u0[0]*s[1] - u0[1]*s[0];

    for(int i=0;i<3;i++) {
      u_p[i] = up[i] + dt_2*E_p[i];
      }

    g_p = sqrt( 1. + u_p[0]*u_p[0] + u_p[1]*u_p[1] + u_p[2]*u_p[2] );

    px[ip] = u_p[0];
    py[ip] = u_p[1];
    pz[ip] = u_p[2];
    g_inv[ip] = 1./g_p;

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
  __constant double *dt_2,
  __constant uint *num_p)
{
  uint ip = get_global_id(0);
  if (ip < *num_p)
   {
    double dt_2_g = (*dt_2) * g_inv[ip];

    double dx = px[ip] * dt_2_g;
    double dy = py[ip] * dt_2_g;
    double dz = pz[ip] * dt_2_g;

    x[ip] += dx;
    y[ip] += dy;
    z[ip] += dz;
   }
}


////////////////////////////////////////////////////////////////////
///////////////////TEST KERNELS ////////////////////////////////////
////////////////////////////////////////////////////////////////////

// Copy sorted particle data of double-type to a new array
// using the shared memory: no speed-up expected, just to see how it works
__kernel void data_align_dbl_tst(
  __global double *x,
  __global double *x_new,
  __global uint *sorted_indx,
  uint num_p)
{
  __local double x_loc[BLOCK_SIZE];

  int base_idx = get_group_id(0)*BLOCK_SIZE;
  int ip_loc = get_local_id(0);
  int ip_glob = base_idx + ip_loc;

  if (ip_glob < num_p)
  {
    x_loc[ip_loc] = x[sorted_indx[ip_glob]];
    x_new[ip_glob] = x_loc[ip_loc];
  }
}

// Another sorting kernel -- something is wrong
__kernel void index_compare_sort(
  __global uint *sorting_index,
  __global uint *sum_out,
  __global uint *new_indx_in_cell,
  __global uint *new_cell_offset,
  __global uint *sum_in_cell,
  __constant uint *Nxm1Nrm1,
  __constant uint *Np)
{
  uint ip = get_global_id(0);

  if (ip < *Np)
   {
    uint new_i_cell = new_indx_in_cell[ip];
    uint new_offset_end = new_cell_offset[*Nxm1Nrm1];
    uint indx_loc;
    uint new_offset_loc;

    if (new_i_cell<*Nxm1Nrm1)
     {
      new_offset_loc = new_cell_offset[new_i_cell];
      indx_loc = atom_add(&sum_in_cell[new_i_cell],1U);
      sorting_index[ip] = new_offset_loc + indx_loc;
     }
    else
     {
      indx_loc = atom_add(&*sum_out, 1U);
      sorting_index[ip] = new_offset_end + indx_loc;
     }
   }
}
