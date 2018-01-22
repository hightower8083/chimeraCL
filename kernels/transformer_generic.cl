// this is a source of transformer kernels for chimeraCL project

__kernel void get_m1(
  __global double2 *fld_m_m1,
  __global double2 *fld_m_p1,
  __constant uint *Nx,
  __constant uint *NxNrm1)
{
  uint i_grid = (uint) get_global_id(0);
  if (i_grid < *NxNrm1)
   {
    uint Nx_loc = *Nx;
    uint ir = i_grid/Nx_loc;
    uint ix = i_grid - ir*Nx_loc;

    uint ix_orig = Nx_loc - ix;
    if (ix == 0) {ix_orig = 0;}

    uint i_grid_orig = ix_orig + ir*Nx_loc;

    fld_m_m1[i_grid].s0 = -fld_m_p1[i_grid_orig].s0;
    fld_m_m1[i_grid].s1 =  fld_m_p1[i_grid_orig].s1;
   }
}


// Get the transofmed fields phase (+) from frame origin X-coordiante
__kernel void get_phase_plus(
  __global double2 *phs_shft,
  __global double *kx,
           double x0,
           uint Nx)
{
  uint ix = (uint) get_global_id(0);
  if (ix < Nx)
   {
    phs_shft[ix].s0 = cos(x0*kx[ix]);
    phs_shft[ix].s1 = sin(x0*kx[ix]);
   }
}

// Get the transofmed fields phase (-) from frame origin X-coordiante
__kernel void get_phase_minus(
  __global double2 *phs_shft,
  __global double *kx,
           double x0,
           uint Nx)
{
  uint ix = (uint) get_global_id(0);
  if (ix < Nx)
   {
    phs_shft[ix].s0 =  cos(x0*kx[ix]);
    phs_shft[ix].s1 = -sin(x0*kx[ix]);
   }
}

// Multiply transofmed fields by phase along X axis
__kernel void multiply_by_phase(
  __global double2 *arr,
  __constant uint *NxNr,
  __constant uint *Nx,
  __global double2 *phs_shft)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *NxNr)
   {
    uint Nx_loc = *Nx;
    uint ir = i_cell/Nx_loc;
    uint ix = i_cell - ir*Nx_loc;

    double arr_re = arr[i_cell].s0;
    double arr_im = arr[i_cell].s1;

    double phs_re = phs_shft[ix].s0;
    double phs_im = phs_shft[ix].s1;

    arr[i_cell].s0 = (arr_re*phs_re - arr_im*phs_im);
    arr[i_cell].s1 = (arr_re*phs_im + arr_im*phs_re);
   }
}
