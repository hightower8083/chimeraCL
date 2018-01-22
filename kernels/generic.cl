// this is a source of generic kernels for chimeraCL project

// Set all elements of complex type field to some value
__kernel void set_cdouble_to(
  __global double2 *x,
           double2 val,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    x[i_cell].s0 = val.s0;
    x[i_cell].s1 = val.s1;
   }
}

// Append a complex type array
__kernel void append_c2c(
  __global double2 *arr_base,
  __global double2 *arr_add,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    arr_base[i_cell].s0 = arr_base[i_cell].s0 + arr_add[i_cell].s0;
    arr_base[i_cell].s1 = arr_base[i_cell].s1 + arr_add[i_cell].s1;
   }
}

// Kernel for z + a*x = z with complex-types
__kernel void zpaxz_c2c(
           double2 a,
  __global double2 *x,
  __global double2 *z,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    z[i_cell].s0 = z[i_cell].s0 + a.s0 * x[i_cell].s0 - a.s1 * x[i_cell].s1;
    z[i_cell].s1 = z[i_cell].s1 + a.s0 * x[i_cell].s1 + a.s1 * x[i_cell].s0;
   }
}

// Kernel for z + a*x = z with complex-types
__kernel void mult_elementwise_d2c(
  __global double  *x,
  __global double2 *z,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    z[i_cell].s0 = x[i_cell] * z[i_cell].s0;
    z[i_cell].s1 = x[i_cell] * z[i_cell].s1;
   }
}

// Kernel for a*x + b*y = z with complex-types
__kernel void axpbyz_c2c(
           double2 a,
  __global double2 *x,
           double2 b,
  __global double2 *y,
  __global double2 *z,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    z[i_cell].s0 = a.s0 * x[i_cell].s0 - a.s1 * x[i_cell].s1 + \
                   b.s0 * y[i_cell].s0 - b.s1 * y[i_cell].s1 ;
    z[i_cell].s1 = a.s0 * x[i_cell].s1 + a.s1 * x[i_cell].s0 + \
                   b.s0 * y[i_cell].s1 + b.s1 * y[i_cell].s0;
   }
}


//  Kernel for a * (b.x) = z with complex-types
// dot product here is over the second index Nx
__kernel void ab_dot_x(
           double2 a,
  __global double *b,
  __global double2 *x,
  __global double2 *z,
           uint NxNr,
           uint Nx)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < NxNr)
   {
    uint ir = i_cell/Nx;
    uint ix = i_cell - ir*Nx;

    z[i_cell].s0 = b[ix]*(a.s0*x[i_cell].s0 - a.s1*x[i_cell].s1);
    z[i_cell].s1 = b[ix]*(a.s0*x[i_cell].s1 + a.s1*x[i_cell].s0);
   }
}

// Cast a double-type array to a complex-type one
__kernel void cast_array_d2c(
  __global double2 *arr_in,
  __global double *arr_out,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    arr_out[i_cell] = arr_in[i_cell].s0;
   }
}
