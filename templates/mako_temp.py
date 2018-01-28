M = 3

mrange = range(1, M+1)

init_modes = ',\n'.join([
'  __global double2 *scl_m{}'.format(m) for m in mrange])


private_cells_alloc = '\n'.join([
'    double scl_cell_m{0}[2][2][2];'.format(m) for m in mrange])

private_cells_init = '\n'.join([
'        scl_cell_m{0}[i][j][0] = 0;'.format(m) for m in mrange])

init_exp = '\n'.join([
'    double exp_m{}[2];'.format(m) for m in mrange])

private_cells_compute_re = '\n'.join([
'          scl_cell_m{0}[i][j][0] += C_cell[i][j]*exp_m{0}[0];'.format(m) for m in mrange])

private_cells_compute_im = '\n'.join([
'          scl_cell_m{0}[i][j][1] += C_cell[i][j]*exp_m{0}[1];'.format(m) for m in mrange])

copy_to_global = '\n'.join([
"""        scl_m{0}[i_dep] = scl_m{0}[i_dep] + (double2) {{scl_cell_m{0}[i][j][0],
                                                   scl_cell_m{0}[i][j][1]}};
""".format(m) for m in mrange])

args_depos = {
'init_modes': init_modes,
'private_cells_alloc': private_cells_alloc,
'private_cells_init': private_cells_init,
'init_exp': init_exp,
'private_cells_compute_re': private_cells_compute_re,
'private_cells_compute_im': private_cells_compute_im,
'copy_to_global': copy_to_global,}
