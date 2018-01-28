
/// this is a source of grid kernels for chimeraCL project

// Depose particles "weights" onto 2D grid via linear projection,
// using groups of 4 cells and barriered steps for each cell
// in a group by calling kernel with different offsets: (0,1,2,3)
// _______________________
//|     |     |     |     |
//|  2  |  3  |  2  |  3  |
//|_____|_____|_____|_____|
//|     |     |     |     |
//|  0  |  1  |  0  |  1  |
//|_____|_____|_____|_____|
//| ____|____ |     |     |
//|| 2  |  3 ||  2  |  3  |
//||____|____||_____|_____|
//||    |    ||     |     |
//||_0__|__1_||  0  |  1  |
//|_____|_____|_____|_____|
//
__kernel void depose_scalar(
           uint cell_offset,
  __global uint *sorting_indx,
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *w,
  __global uint *indx_offset,
             char charge,
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv,
  __constant uint *NxNr_4,
  __global double *scl_m0,
${init_modes})
{
  // running kernels over the 4cells
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *NxNr_4)
   {
    // get numbers of cells and period of 4cell-grid
    uint Nx_grid = *Nx;
    uint Nx_cell = Nx_grid-1;
    uint Nx_2 = Nx_grid/2;
    uint Nr_cell = *Nr-1;

    // get indicies of 4cell origin (left-bottom)
    uint ir = i_cell/Nx_2;
    uint ix = i_cell - ir*Nx_2;

    // convert 4cell indicies to global grid
    ix *= 2;
    ir *= 2;

    // apply offset whithin a 4cell
    if (cell_offset==1)
      {ix += 1;}
    else if (cell_offset==2)
      {ir += 1;}
    else if (cell_offset==3)
      {ix += 1;ir += 1;}

if (ix>0 && ix<Nx_cell-1 && ir<Nr_cell-1 ){
    // get 1D indicies of the selected
    // cell and grid node on the global grid
    uint i_cell_glob = ix + ir*Nx_cell;
    uint i_grid_glob = ix + ir*Nx_grid;

    // get particles indicies in the selected cell
    uint ip_start = indx_offset[i_cell_glob];
    uint ip_end = indx_offset[i_cell_glob+1];

    // skip empty cells
if (ip_start != ip_end){

    // allocate few integer counters
    uint i,j,i_dep;

    // allocate privite cell array for the deposition
    double scl_cell_m0[2][2];
${private_cells_alloc}

    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        scl_cell_m0[i][j] = 0;
${private_cells_init}
        scl_cell_m1[i][j][0] = 0;
        scl_cell_m1[i][j][1] = 0;
        }}

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double C_cell[2][2];
${init_exp}

    double xp, yp, zp, wp, rp, rp_inv;
    double rmin_loc = *rmin;
    double dr_inv_loc = *dr_inv;
    double xmin_loc = *xmin;
    double dx_inv_loc = *dx_inv;
    uint ip_srtd;

    // run over the particles for linear deposition
    for (uint ip=ip_start; ip<ip_end; ip++){

      ip_srtd = sorting_indx[ip];
      xp = x[ip_srtd];
      yp = y[ip_srtd];
      zp = z[ip_srtd];
      wp = w[ip_srtd]*charge;

      rp = sqrt(yp*yp + zp*zp);

      rp_inv = 1./rp;
      exp_m1[0] = yp*rp_inv;
      exp_m1[1] = zp*rp_inv;

      sX1 = ( xp - xmin_loc )*dx_inv_loc - ix;
      sX0 = 1.0 - sX1;
      sR1 = ( rp - rmin_loc )*dr_inv_loc - ir;
      sR0 = 1.0 - sR1;

      sX0 *= wp;
      sX1 *= wp;

      C_cell[0][0] = sR0*sX0;
      C_cell[0][1] = sR0*sX1;
      C_cell[1][0] = sR1*sX0;
      C_cell[1][1] = sR1*sX1;

      for (i=0;i<2;i++){
        for (j=0;j<2;j++){
          scl_cell_m0[i][j] += C_cell[i][j];
${private_cells_compute_re}
${private_cells_compute_im}
          }}
    }

    // write to the global field memory
    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        i_dep = i_grid_glob + j + Nx_grid*i;
        scl_m0[i_dep] = scl_m0[i_dep] + scl_cell_m0[i][j];
${copy_to_global}
        }}
  }
}}
}
