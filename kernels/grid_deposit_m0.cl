// this is a source of grid kernels for chimeraCL project

// Depose particles "weights" onto 2D grid via linear projection,
// using groups of 4 cells and barrierd steps for each cell
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
  __global double *scl_m0)
{
  // running kernels over the 4cells
  uint i_cell = (uint) get_global_id(0);
if (i_cell < *NxNr_4)
   {
    // get cell number and period of 4cell-grid
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

if (ix<Nx_cell && ir<Nr_cell ){

    // get 1D indicies of the selected
    // cell and grid node on the global grid
    uint i_cell_glob = ix + ir*Nx_cell;
    uint i_grid_glob = ix + ir*Nx_grid;

    // get particles indicies in the selected cell
    uint ip_start = indx_offset[i_cell_glob];
    uint ip_end = indx_offset[i_cell_glob+1];

    // skip empty cells
if (ip_start != ip_end) {

    // allocate privite cell array for the deposition
    double scl_cell_m0[2][2] = {{0.,0.},{0.,0.}};
    double C_cell[2][2];

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;

    double xp, yp, zp, wp,rp;
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
      wp = w[ip_srtd] * charge;

      rp = sqrt(yp*yp + zp*zp);

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

      for (int i=0;i<2;i++){
        for (int j=0;j<2;j++)
          {scl_cell_m0[i][j] += C_cell[i][j];}}
    }

    // write to the global field memory
    for (int i=0;i<2;i++){
      for (int j=0;j<2;j++){
        int i_dep = i_grid_glob + j + Nx_grid*i;
        scl_m0[i_dep] += scl_cell_m0[i][j];
      }}
  }
}}}

// Linear projection of a scalar onto 2D grid
__kernel void project_scalar(
  __global uint *sorting_indx,
  __global double *scl_proj,
  __global double *x,
  __global double *y,
  __global double *z,
  __global uint *indx_offset,
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv,
  __constant uint *Nxm1Nrm1,
  __global double *scl_m0)
{
  // running kernels over the cells
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *Nxm1Nrm1)
   {
    // get cell number and period of grid
    uint Nx_grid = *Nx;
    uint Nx_cell = Nx_grid-1;

    // get indicies of selected cell
    uint ir = i_cell/Nx_cell;
    uint ix = i_cell - ir*Nx_cell;

    // get 1D indicies of the selected
    // cell and grid node on the global grid
    uint i_cell_glob = ix + ir*Nx_cell;
    uint i_grid_glob = ix + ir*Nx_grid;

    // get particles indicies in the selected cell
    uint ip_start = indx_offset[i_cell_glob];
    uint ip_end = indx_offset[i_cell_glob+1];

    // skip empty cells
    if (ip_start == ip_end) {return;}

    // allocate privite cell array for the deposition
    double scl_cell_m0[2][2] ;

    for (int i=0;i<2;i++){
      for (int j=0;j<2;j++){
        int i_dep = i_grid_glob + j + Nx_grid*i;
        scl_cell_m0[i][j] = scl_m0[i_dep];
      }}

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double C_cell[2][2];

    double xp, yp, zp, rp,scl_proj_p;
    double rmin_loc = *rmin;
    double dr_inv_loc = *dr_inv;
    double xmin_loc = *xmin;
    double dx_inv_loc = *dx_inv;
    uint ip_srtd;

    // run over the particles for linear deposition
    for (uint ip=ip_start; ip<ip_end; ip++)
     {
      ip_srtd = sorting_indx[ip];
      xp = x[ip_srtd];
      yp = y[ip_srtd];
      zp = z[ip_srtd];

      rp = sqrt(yp*yp + zp*zp);

      scl_proj_p = 0;

      sX1 = ( xp - xmin_loc )*dx_inv_loc - ix;
      sX0 = 1.0 - sX1;
      sR1 = ( rp - rmin_loc )*dr_inv_loc - ir;
      sR0 = 1.0 - sR1;

      C_cell[0][0] = sR0*sX0;
      C_cell[0][1] = sR0*sX1;
      C_cell[1][0] = sR1*sX0;
      C_cell[1][1] = sR1*sX1;

      for (int i=0;i<2;i++){
        for (int j=0;j<2;j++){
          scl_proj_p += C_cell[i][j]*scl_cell_m0[i][j];
        }}

      scl_proj[ip_srtd] += scl_proj_p ;
     }
   }
}

// Depose weighted particles vectors onto 2D grid via linear projection,
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
__kernel void depose_vector(
           uint cell_offset,
  __global uint *sorting_indx,
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *ux,
  __global double *uy,
  __global double *uz,
  __global double *g_inv,
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
  __global double *vec_x_m0,
  __global double *vec_y_m0,
  __global double *vec_z_m0)
{
  // running kernels over the 4cells
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *NxNr_4)
   {
    // get cell number and period of 4cell-grid
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

if (ix<Nx_cell && ir<Nr_cell ){
    // get 1D indicies of the selected
    // cell and grid node on the global grid
    uint i_cell_glob = ix + ir*Nx_cell;
    uint i_grid_glob = ix + ir*Nx_grid;

    // get particles indicies in the selected cell
    uint ip_start = indx_offset[i_cell_glob];
    uint ip_end = indx_offset[i_cell_glob+1];

    // skip empty cells
if (ip_start != ip_end){

    // allocate privite cell array for the deposition
    double vec_cell_m0[3][2][2];

    for (int k=0; k<3; k++){
      for (int i=0; i<2; i++){
        for (int j=0; j<2; j++){
          vec_cell_m0[k][i][j] = 0;
        }}}

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double C_cell[2][2];

    double xp, yp, zp, wp,rp;
    double jp[3];
    double rmin_loc = *rmin;
    double dr_inv_loc = *dr_inv;
    double xmin_loc = *xmin;
    double dx_inv_loc = *dx_inv;
    uint ip_srtd;

    // run over the particles for linear deposition
    for (uint ip=ip_start; ip<ip_end; ip++)
     {
      ip_srtd = sorting_indx[ip];
      xp = x[ip_srtd];
      yp = y[ip_srtd];
      zp = z[ip_srtd];
      jp[0] = ux[ip_srtd];
      jp[1] = uy[ip_srtd];
      jp[2] = uz[ip_srtd];
      wp = w[ip_srtd] * g_inv[ip_srtd] * charge;

      rp = sqrt(yp*yp + zp*zp);

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

      for (int k=0;k<3;k++){
        for (int i=0;i<2;i++){
          for (int j=0;j<2;j++){
            vec_cell_m0[k][i][j] += C_cell[i][j] * jp[k];
          }}}
     }

    // write to the global field memory

    for (int i=0;i<2;i++){
      for (int j=0;j<2;j++){
        int i_dep = i_grid_glob + j + Nx_grid*i;
        vec_x_m0[i_dep] += vec_cell_m0[0][i][j];
        vec_y_m0[i_dep] += vec_cell_m0[1][i][j];
        vec_z_m0[i_dep] += vec_cell_m0[2][i][j];
      }}
  }
}}
}

// Linear projection of a weighted vector of particles onto 2D grid
__kernel void project_vec6(
  __global uint *sorting_indx,
  __global double *vec_x_1_proj,
  __global double *vec_y_1_proj,
  __global double *vec_z_1_proj,
  __global double *vec_x_2_proj,
  __global double *vec_y_2_proj,
  __global double *vec_z_2_proj,
  __global double *x,
  __global double *y,
  __global double *z,
  __global uint *indx_offset,
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv,
  __constant uint *Nxm1Nrm1,
  __global double *vec_x_1_m0,
  __global double *vec_y_1_m0,
  __global double *vec_z_1_m0,
  __global double *vec_x_2_m0,
  __global double *vec_y_2_m0,
  __global double *vec_z_2_m0)
{
  // running kernels over the cells
  uint i_cell = (uint) get_global_id(0);
  if (i_cell < *Nxm1Nrm1)
   {
    // get cell number and period of grid
    uint Nx_grid = *Nx;
    uint Nx_cell = Nx_grid-1;

    // get indicies of cell
    uint ir = i_cell/Nx_cell;
    uint ix = i_cell - ir*Nx_cell;

    // get 1D indicies of the selected
    // cell and grid node on the global grid
    uint i_cell_glob = ix + ir*Nx_cell;
    uint i_grid_glob = ix + ir*Nx_grid;

    // get particles indicies in the selected cell
    uint ip_start = indx_offset[i_cell_glob];
    uint ip_end = indx_offset[i_cell_glob+1];

    // skip empty cells
    if (ip_start == ip_end) {return;}

    // allocate privite cell array for the deposition
    double vec_1_cell_m0[3][2][2] ;
    double vec_2_cell_m0[3][2][2] ;

    for (int i=0;i<2;i++){
      for (int j=0;j<2;j++){
        int i_dep = i_grid_glob + j + Nx_grid*i;
        vec_1_cell_m0[0][i][j] = vec_x_1_m0[i_dep];
        vec_1_cell_m0[1][i][j] = vec_y_1_m0[i_dep];
        vec_1_cell_m0[2][i][j] = vec_z_1_m0[i_dep];

        vec_2_cell_m0[0][i][j] = vec_x_2_m0[i_dep];
        vec_2_cell_m0[1][i][j] = vec_y_2_m0[i_dep];
        vec_2_cell_m0[2][i][j] = vec_z_2_m0[i_dep];
      }}

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double C_cell[2][2];

    double vec_1_p[3];
    double vec_2_p[3];

    double xp, yp, zp, rp;
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

      rp = sqrt(yp*yp + zp*zp);

      for (int k=0;k<3;k++){
        vec_1_p[k] = 0;
        vec_2_p[k] = 0;
      }

      sX1 = ( xp - xmin_loc )*dx_inv_loc - ix;
      sX0 = 1.0 - sX1;
      sR1 = ( rp - rmin_loc )*dr_inv_loc - ir;
      sR0 = 1.0 - sR1;

      C_cell[0][0] = sR0*sX0;
      C_cell[0][1] = sR0*sX1;
      C_cell[1][0] = sR1*sX0;
      C_cell[1][1] = sR1*sX1;

      for (int k=0;k<3;k++){
        for (int i=0;i<2;i++){
          for (int j=0;j<2;j++){
            vec_1_p[k] += C_cell[i][j]*vec_1_cell_m0[k][i][j];
            vec_2_p[k] += C_cell[i][j]*vec_2_cell_m0[k][i][j];
          }}}

      vec_x_1_proj[ip_srtd] += vec_1_p[0] ;
      vec_y_1_proj[ip_srtd] += vec_1_p[1] ;
      vec_z_1_proj[ip_srtd] += vec_1_p[2] ;

      vec_x_2_proj[ip_srtd] += vec_2_p[0] ;
      vec_y_2_proj[ip_srtd] += vec_2_p[1] ;
      vec_z_2_proj[ip_srtd] += vec_2_p[2] ;
    }
  }
}
