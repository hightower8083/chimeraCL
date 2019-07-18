// this is a source of grid kernels for chimeraCL project

// Divide double type 2D array by R along 0-th axis
__kernel void divide_by_dv_d(
  __global double *arr,
  __constant uint *NxNr,
  __constant uint *Nx,
  __global double *dv_inv)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *NxNr)
   {
    uint Nx_loc = *Nx;
    uint ir = i_cell/Nx_loc;
    arr[i_cell] *= dv_inv[ir];
   }
}

// Divide complex type 2D array by R along 0-th axis
__kernel void divide_by_dv_c(
  __global double2 *arr,
  __constant uint *NxNr,
  __constant uint *Nx,
  __global double *dv_inv)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *NxNr)
   {
    uint Nx_loc = *Nx;
    uint ir = i_cell/Nx_loc ;
    arr[i_cell].s0 *= dv_inv[ir];
    arr[i_cell].s1 *= dv_inv[ir];
   }
}

// Substract 0-th row from 1-st for double type 2D array
__kernel void treat_axis_d(
  __global double *arr,
             uint Nx)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < Nx)
   {
    arr[i_cell + Nx] -= arr[i_cell];
   }
}

// Substract 0-th row from 1-st for complex type 2D array
__kernel void treat_axis_c(
  __global double2 *arr,
               uint Nx)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < Nx)
   {
    arr[i_cell + Nx].s0 -= arr[i_cell].s0;
    arr[i_cell + Nx].s1 -= arr[i_cell].s1;
   }
}

// Fill the first raw of field array (index=-1/2):
// symmetric for m=0
__kernel void warp_axis_m0_d(
  __global double *arr,
               uint Nx)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < Nx)
   {
    arr[i_cell] = arr[i_cell + Nx];
   }
}

// Fill the first raw of field array (index=-1/2):
// antisymmetric for m=1, to impose Jm(0)=0 for m>0
__kernel void warp_axis_m1plus_c(
  __global double2 *arr,
               uint Nx)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < Nx)
   {
    arr[i_cell].s0 = -arr[i_cell + Nx].s0;
    arr[i_cell].s1 = -arr[i_cell + Nx].s1;
   }
}
