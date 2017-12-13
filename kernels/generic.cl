// this is a source of generic kernels for chimeraCL project

// Set to zero all elements of complex type field
__kernel void set_cdouble_to(
  __global double2 *rho,
           double2 val,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    rho[i_cell].s0 = val.s0;
    rho[i_cell].s1 = val.s1;
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

// Kernel for a*x+b*y with complex-types
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
