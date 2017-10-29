// this is a source of generic kernels for chimeraCL project

// Set to zero all elements of complex type field
__kernel void set_cdouble_to_zero(
  __global double2 *rho,
           uint arr_size)
{
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < arr_size)
   {
    rho[i_cell].s0 = 0.0;
    rho[i_cell].s1 = 0.0;
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
