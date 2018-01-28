import numpy as np
from mako.template import Template

head_init = {
  'init_scl': '  __global double2 *scl_m{}',
  'init_vec': """
  __global double2 *vec_x_m{0},
  __global double2 *vec_y_m{0},
  __global double2 *vec_z_m{0}""",
  'init_eb': """
  __global double2 *ex_m{0},
  __global double2 *ey_m{0},
  __global double2 *ez_m{0},
  __global double2 *bx_m{0},
  __global double2 *by_m{0},
  __global double2 *bz_m{0}"""
}

generic = {
  'vec_cells_alloc'     : '    double vec_cell_m{}[3][2][2][2]',
  'scl_cells_alloc'     : '    double scl_cell_m{0}[2][2][2];',
  'eb_cells_alloc'      : """
    double e_cell_m{0}[3][2][2][2] ;
    double b_cell_m{0}[3][2][2][2] ;""",
  'vec_cells_init'      : """
          vec_cell_m{0}[k][i][j][0] = 0.;
          vec_cell_m{0}[k][i][j][1] = 0.;""",
  'scl_cells_init'      : """
        scl_cell_m{0}[i][j][0] = 0.;
        scl_cell_m{0}[i][j][1] = 0.;""",

  'eb_cells_init'       : """
        e_cell_m{0}[0][i][j][0] = 2 * ((double) ex_m{0}[i_loc].s0);
        e_cell_m{0}[0][i][j][1] = 2 * ((double) ex_m{0}[i_loc].s1);

        e_cell_m{0}[1][i][j][0] = 2 * ((double) ey_m{0}[i_loc].s0);
        e_cell_m{0}[1][i][j][1] = 2 * ((double) ey_m{0}[i_loc].s1);

        e_cell_m{0}[2][i][j][0] = 2 * ((double) ez_m{0}[i_loc].s0);
        e_cell_m{0}[2][i][j][1] = 2 * ((double) ez_m{0}[i_loc].s1);

        b_cell_m{0}[0][i][j][0] = 2 * ((double) bx_m{0}[i_loc].s0);
        b_cell_m{0}[0][i][j][1] = 2 * ((double) bx_m{0}[i_loc].s1);

        b_cell_m{0}[1][i][j][0] = 2 * ((double) by_m{0}[i_loc].s0);
        b_cell_m{0}[1][i][j][1] = 2 * ((double) by_m{0}[i_loc].s1);

        b_cell_m{0}[2][i][j][0] = 2 * ((double) bz_m{0}[i_loc].s0);
        b_cell_m{0}[2][i][j][1] = 2 * ((double) bz_m{0}[i_loc].s1);""",
  'init_exp'            : '    double exp_m{}[2];',
  'vec_cells_compute': """
            vec_cell_m{0}[k][i][j][0] += jp_proj*exp_m{0}[0];
            vec_cell_m{0}[k][i][j][1] += jp_proj*exp_m{0}[1];""",
  'scl_cells_compute': """
          scl_cell_m{0}[i][j][0] += C_cell[i][j]*exp_m{0}[0];
          scl_cell_m{0}[i][j][1] += C_cell[i][j]*exp_m{0}[1];""",
  'eb_cells_compute' : """
          e_p[k] += C_cell[i][j]*e_cell_m{0}[k][i][j][0]*exp_m{0}[0];
          e_p[k] -= C_cell[i][j]*e_cell_m{0}[k][i][j][1]*exp_m{0}[1];
          b_p[k] += C_cell[i][j]*b_cell_m{0}[k][i][j][0]*exp_m{0}[0];
          b_p[k] -= C_cell[i][j]*b_cell_m{0}[k][i][j][1]*exp_m{0}[1];""",
  'copy_vec_to_global'  : """
        vec_x_m{0}[i_dep] = vec_x_m{0}[i_dep] + (double2) {{vec_cell_m{0}[0][i][j][0],
                                                          vec_cell_m{0}[0][i][j][1]}};
        vec_y_m{0}[i_dep] = vec_y_m{0}[i_dep] + (double2) {{vec_cell_m{0}[1][i][j][0],
                                                          vec_cell_m{0}[1][i][j][1]}};
        vec_z_m{0}[i_dep] = vec_z_m{0}[i_dep] + (double2) {{vec_cell_m{0}[2][i][j][0],
                                                          vec_cell_m{0}[2][i][j][1]}};""",
  'copy_scl_to_global'  : """
        scl_m{0}[i_dep] = scl_m{0}[i_dep] + (double2) {{scl_cell_m{0}[i][j][0],
                                                   scl_cell_m{0}[i][j][1]}};"""
}

cross_recursive = {
  'exp_compute_dep': (
    '      exp_m{}[0] = yp*rp_inv;',
    '      exp_m{}[1] = zp*rp_inv;',
    '      exp_m{0}[0] = exp_m{1}[0]*exp_m1[0] - exp_m{1}[1]*exp_m1[1];',
    '      exp_m{0}[1] = exp_m{1}[0]*exp_m1[1] + exp_m{1}[1]*exp_m1[0];'
                     )
  'exp_compute_gath': (
    '      exp_m{}[0] = yp*rp_inv;',
    '      exp_m{}[1] = -zp*rp_inv;',
    '      exp_m{0}[0] = exp_m{1}[0]*exp_m1[0] - exp_m{1}[1]*exp_m1[1];',
    '      exp_m{0}[1] = exp_m{1}[0]*exp_m1[1] + exp_m{1}[1]*exp_m1[0];'
                     )
}

def generate_code(M, head_init = {}, generic={}, cross_recursive={}):

    grid_kernel_template = Template(filename='grid_kernel_template.cl')

    Args = {}

    mrange = range(1, M+1)

    for key in head_init.keys():
        Args[key] = ',\n'.join([head_init[key].format(m) for m in mrange])
        print(Args[key])

    for key in generic.keys():
        Args[key] = '\n'.join([generic[key].format(m) for m in mrange])

    for key in cross_recursive.keys():
        arg_0 = []
        arg_1 = []

        if M>0:
            arg_0.append(cross_recursive[key][0].format(1))
            arg_1.append(cross_recursive[key][1].format(1))

        for m in range(2,M+1):
            arg_0.append(cross_recursive[key][2].format(m,m-1))
            arg_1.append(cross_recursive[key][3].format(m,m-1))

        arg = list(np.vstack((np.array(arg_0),
                              np.array(arg_1) )).T.ravel())

        Args[key] = '\n'.join(arg)
        src = grid_kernel_template.render(**Args)

    return src

