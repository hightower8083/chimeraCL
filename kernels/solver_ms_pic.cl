// this is a source of maxwell solver kernels for chimeraCL project

__kernel void advance_e_g_m(
  __constant uint *NxNr,
  __constant double *dt_inv,
  __global double *c1_m,
  __global double *c2_m,
  __global double *c3_m,
  __global double2 *e_x_m,
  __global double2 *e_y_m,
  __global double2 *e_z_m,
  __global double2 *g_x_m,
  __global double2 *g_y_m,
  __global double2 *g_z_m,
  __global double2 *j_x_m,
  __global double2 *j_y_m,
  __global double2 *j_z_m,
  __global double2 *n0_x_m,
  __global double2 *n0_y_m,
  __global double2 *n0_z_m,
  __global double2 *n1_x_m,
  __global double2 *n1_y_m,
  __global double2 *n1_z_m)
{
  uint i_grid = (uint) get_global_id(0);
  if (i_grid < *NxNr)
   {
    double e0[3][2] = {{e_x_m[i_grid].s0, e_x_m[i_grid].s1},
                       {e_y_m[i_grid].s0, e_y_m[i_grid].s1},
                       {e_z_m[i_grid].s0, e_z_m[i_grid].s1}
                      };

    double g0[3][2] = {{g_x_m[i_grid].s0, g_x_m[i_grid].s1},
                       {g_y_m[i_grid].s0, g_y_m[i_grid].s1},
                       {g_z_m[i_grid].s0, g_z_m[i_grid].s1}
                      };

    double j0[3][2] = {{j_x_m[i_grid].s0, j_x_m[i_grid].s1},
                       {j_y_m[i_grid].s0, j_y_m[i_grid].s1},
                       {j_z_m[i_grid].s0, j_z_m[i_grid].s1}
                      };

    double n0[3][2] = {{n0_x_m[i_grid].s0, n0_x_m[i_grid].s1},
                       {n0_y_m[i_grid].s0, n0_y_m[i_grid].s1},
                       {n0_z_m[i_grid].s0, n0_z_m[i_grid].s1}
                      };

    double n1[3][2] = {{n1_x_m[i_grid].s0, n1_x_m[i_grid].s1},
                       {n1_y_m[i_grid].s0, n1_y_m[i_grid].s1},
                       {n1_z_m[i_grid].s0, n1_z_m[i_grid].s1}
                      };

    double c1 = c1_m[i_grid];
    double c2 = c2_m[i_grid];
    double c3 = c3_m[i_grid];

    double dt_inv_loc = *dt_inv;
    double e1[3][2], g1[3][2];

    for (int k=0;k<3;k++){
        for (int i=0;i<2;i++){
            e1[k][i] = c1*e0[k][i] + c2*c3*(g0[k][i]-j0[k][i]) +
              c3*(c1*n0[k][i] - n1[k][i] -
                  (n0[k][i]-n1[k][i]) * dt_inv_loc * c2 * c3);

            g1[k][i] = -c2*e0[k][i] + c1*(g0[k][i]-j0[k][i]) + j0[k][i] +
              c3*(dt_inv_loc*(1.-c1)*(n0[k][i]-n1[k][i]) - c2*n0[k][i]) ;
        }
    }
    e_x_m[i_grid] = (double2) {e1[0][0], e1[0][1]};
    e_y_m[i_grid] = (double2) {e1[1][0], e1[1][1]};
    e_z_m[i_grid] = (double2) {e1[2][0], e1[2][1]};

    g_x_m[i_grid] = (double2) {g1[0][0], g1[0][1]};
    g_y_m[i_grid] = (double2) {g1[1][0], g1[1][1]};
    g_z_m[i_grid] = (double2) {g1[2][0], g1[2][1]};
   }
}
