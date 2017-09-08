// this is a source of grid kernels for chimeraCL project
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

// Set to zero all elements of complex type field
__kernel void set_cdouble_to_zero(
  __global cdouble_t *rho,
  __constant uint *NxNr)
{
  uint i_cell = get_global_id(0);
  if (i_cell < *NxNr)
   {
    rho[i_cell].real = 0.0;
    rho[i_cell].imag = 0.0;
   }
}

// Divide double type 2D array by R along 0-th axis
__kernel void divide_by_r_dbl(
  __global double *arr,
  __constant uint *NxNr,
  __constant uint *Nx,
  __global double *r_inv)
{
  uint i_cell = get_global_id(0);
  if (i_cell < *NxNr)
   {
    uint Nx_loc = *Nx;
    uint ir = i_cell/Nx_loc;
    arr[i_cell] *= r_inv[ir];
   }
}

// Divide complex type 2D array by R along 0-th axis
__kernel void divide_by_r_clx(
  __global cdouble_t *arr,
  __constant uint *NxNr,
  __constant uint *Nx,
  __global double *r_inv)
{
  uint i_cell = get_global_id(0);
  if (i_cell < *NxNr)
   {
    uint Nx_loc = *Nx;
    uint ir = i_cell/Nx_loc ;
    arr[i_cell].real *= r_inv[ir];
    arr[i_cell].imag *= r_inv[ir];
   }
}

// Substract 0-th row from 1-st for double type 2D array
__kernel void treat_axis_dbl(
  __global double *arr,
  __constant uint *Nx)
{
  uint i_cell = get_global_id(0);
  if (i_cell < *Nx)
   {
    uint Nx_loc = *Nx;
    uint ir = i_cell/Nx_loc;
    arr[i_cell + Nx_loc] -= arr[i_cell];
   }
}

// Substract 0-th row from 1-st for complex type 2D array
__kernel void treat_axis_clx(
  __global cdouble_t *arr,
  __constant uint *Nx)
{
  uint i_cell = get_global_id(0);
  if (i_cell < *Nx)
   {
    uint Nx_loc = *Nx;
    uint ir = i_cell/Nx_loc;
    arr[i_cell+Nx_loc].real -= arr[i_cell].real;
    arr[i_cell+Nx_loc].imag -= arr[i_cell].imag;
   }
}

// Cast a double-type array to a complex-type one
__kernel void cast_array_d2c(
  __global cdouble_t *arr_in,
  __global double *arr_out,
  __constant uint *Nxm1Nrm1)
{
  uint i_cell = get_global_id(0);
  if (i_cell < *Nxm1Nrm1)
   {
    arr_out[i_cell] = arr_in[i_cell].real;
   }
}

// Copy a double-type odd-lengths 2D array (Nr Nx)
// to a truncated double-type even-lengths one (Nr-1,Nx-1)
// ____________        __________
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|  -->  |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|
//
__kernel void copy_array_to_even_grid_d2d(
    __global double *arr_in,
    __global double *arr_out,
    __constant uint *Nx,
    __constant uint *Nxm1Nrm1)
{
    uint i_cell_even = get_global_id(0);
    if (i_cell_even < *Nxm1Nrm1)
    {
        uint Nx_grid_odd = *Nx;
        uint Nx_grid_evn = Nx_grid_odd - 1;

        uint ir = i_cell_even/Nx_grid_evn ;
        uint ix = i_cell_even - ir*Nx_grid_evn ;

        uint i_cell_odd = ix + (ir+1)*Nx_grid_odd;

        arr_out[i_cell_even] = arr_in[i_cell_odd];
    }
}

// Copy a double-type odd-lengths 2D array (Nr Nx)
// to a truncated complex-type even-lengths one (Nr-1,Nx-1)
// ____________        __________
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|  -->  |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|
//
__kernel void copy_array_to_even_grid_d2c(
    __global double *arr_in,
    __global cdouble_t *arr_out,
    __constant uint *Nx,
    __constant uint *Nxm1Nrm1)
{
    uint i_cell_even = get_global_id(0);
    if (i_cell_even < *Nxm1Nrm1)
    {
        uint Nx_grid_odd = *Nx;
        uint Nx_grid_evn = Nx_grid_odd - 1;

        uint ir = i_cell_even/Nx_grid_evn ;
        uint ix = i_cell_even - ir*Nx_grid_evn ;

        uint i_cell_odd = ix + (ir+1)*Nx_grid_odd;

        arr_out[i_cell_even].real = arr_in[i_cell_odd];
    }
}

// Copy a complex-type odd-lengths 2D array (Nr Nx)
// to a truncated complex-type even-lengths one (Nr-1,Nx-1)
// ____________        __________
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|  -->  |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|       |_|_|_|_|_|
// |_|_|_|_|_|_|
//
__kernel void copy_array_to_even_grid_c2c(
    __global cdouble_t *arr_in,
    __global cdouble_t *arr_out,
    __constant uint *Nx,
    __constant uint *Nxm1Nrm1)
{
    uint i_cell_even = get_global_id(0);
    if (i_cell_even < *Nxm1Nrm1)
    {
        uint Nx_grid_odd = *Nx;
        uint Nx_grid_evn = Nx_grid_odd - 1;

        uint ir = i_cell_even/Nx_grid_evn ;
        uint ix = i_cell_even - ir*Nx_grid_evn ;

        uint i_cell_odd = ix + (ir+1)*Nx_grid_odd;

        arr_out[i_cell_even].real = arr_in[i_cell_odd].real;
        arr_out[i_cell_even].imag = arr_in[i_cell_odd].imag;
    }
}

// Copy a truncated double-type even-lengths 2D array
// (Nr-1 Nx-1) to a double-type odd-lengths one (Nr,Nx)
// __________        ____________
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_| -->   |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
//                   |_|_|_|_|_|_|
//
__kernel void copy_array_to_odd_grid_d2d(
    __global double *arr_in,
    __global double *arr_out,
    __constant uint *Nx,
    __constant uint *Nxm1Nrm1)
{
    uint i_cell_even = get_global_id(0);
    if (i_cell_even < *Nxm1Nrm1)
    {
        uint Nx_grid_odd = *Nx;
        uint Nx_grid_evn = Nx_grid_odd - 1;

        uint ir = i_cell_even/Nx_grid_evn ;
        uint ix = i_cell_even - ir*Nx_grid_evn ;

        uint i_cell_odd = ix + (ir+1)*Nx_grid_odd;

        arr_out[i_cell_odd] = arr_in[i_cell_even];
    }
}

// Copy a truncated double-type even-lengths 2D array
// (Nr-1 Nx-1) to a complex-type odd-lengths one (Nr,Nx)
// __________        ____________
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_| -->   |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
//                   |_|_|_|_|_|_|
//
__kernel void copy_array_to_odd_grid_d2c(
    __global double *arr_in,
    __global cdouble_t *arr_out,
    __constant uint *Nx,
    __constant uint *Nxm1Nrm1)
{
    uint i_cell_even = get_global_id(0);
    if (i_cell_even < *Nxm1Nrm1)
    {
        uint Nx_grid_odd = *Nx;
        uint Nx_grid_evn = Nx_grid_odd - 1;

        uint ir = i_cell_even/Nx_grid_evn ;
        uint ix = i_cell_even - ir*Nx_grid_evn ;

        uint i_cell_odd = ix + (ir+1)*Nx_grid_odd;

        arr_out[i_cell_odd].real = arr_in[i_cell_even];
    }
}

// Copy a truncated complex-type even-lengths 2D array
// (Nr-1 Nx-1) to a complex-type odd-lengths one (Nr,Nx)
// __________        ____________
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_| -->   |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
// |_|_|_|_|_|       |_|_|_|_|_|_|
//                   |_|_|_|_|_|_|
//
__kernel void copy_array_to_odd_grid_c2c(
    __global cdouble_t *arr_in,
    __global cdouble_t *arr_out,
    __constant uint *Nx,
    __constant uint *Nxm1Nrm1)
{
    uint i_cell_even = get_global_id(0);
    if (i_cell_even < *Nxm1Nrm1)
    {
        uint Nx_grid_odd = *Nx;
        uint Nx_grid_evn = Nx_grid_odd - 1;

        uint ir = i_cell_even/Nx_grid_evn ;
        uint ix = i_cell_even - ir*Nx_grid_evn ;

        uint i_cell_odd = ix + (ir+1)*Nx_grid_odd;

        arr_out[i_cell_odd].real = arr_in[i_cell_even].real;
        arr_out[i_cell_odd].imag = arr_in[i_cell_even].imag;
    }
}
