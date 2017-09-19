// this is a source of generic kernels for chimeraCL project

// Set to zero all elements of complex type field
__kernel void advance_e_g(
  __constant double *dt_inv,
  __constant uint *NxNr,
  __global double *c1_m0,
  __global double *c2_m0,
  __global double *c3_m0,
  __global double *e_x_m0,
  __global double *e_y_m0,
  __global double *e_z_m0,
  __global double *g_x_m0,
  __global double *g_y_m0,
  __global double *g_z_m0,
  __global double *j_x_m0,
  __global double *j_y_m0,
  __global double *j_z_m0,
  __global double *n0_x_m0,
  __global double *n0_y_m0,
  __global double *n0_z_m0,
  __global double *n1_x_m0,
  __global double *n1_y_m0,
  __global double *n1_z_m0)
{
  uint i_grid = get_global_id(0);
  if (i_cell < *NxNr)
   {
    double e0[3] = {e_x_m0[i_grid], e_y_m0[i_grid], e_z_m0[i_grid]};
    double g0[3] = {g_x_m0[i_grid], g_y_m0[i_grid], g_z_m0[i_grid]};
    double j0[3] = {j_x_m0[i_grid], j_y_m0[i_grid], j_z_m0[i_grid]};
    double n0[3] = {n0_x_m0[i_grid], n0_y_m0[i_grid], n0_z_m0[i_grid]};
    double n1[3] = {n1_x_m0[i_grid], n1_y_m0[i_grid], n1_z_m0[i_grid]};

    double c1 = *c1_m0[i_grid];
    double c2 = *c2_m0[i_grid];
    double c3 = *c3_m0[i_grid];

    double dt_inv_loc = *dt_inv;
    double e1[3], g1[3];

    for (int k=0;k<3;k++){
        e1[k] = c1*e0[k] + c2*(g0[k]-j0[k]) +
          c3*(c1*n0[k] + n1[k] - dt_inv_loc*c2*(n0[k]-n1[k]))

        g1[k] = c1*e0[k] + c2*(g0[k]-j0[k]) +
          c3*(c1*n0[k]+n1[k]-dt_inv_loc*c2*(n0[k]-n1[k]))

        }
    e_x_m0[i_grid] = e1[0];
    e_y_m0[i_grid] = e1[1];
    e_z_m0[i_grid] = e1[2];

    g_x_m0[i_grid] = g1[0];
    g_y_m0[i_grid] = g1[1];
    g_z_m0[i_grid] = g1[2];
   }
}
