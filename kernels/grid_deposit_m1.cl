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
  __global double2 *scl_m1)
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

    // allocate few integer counters
    uint i,j,i_dep;

    // allocate privite cell array for the deposition
    double scl_cell_m0[2][2];
    double scl_cell_m1[2][2][2];

    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        scl_cell_m0[i][j] = 0;
        scl_cell_m1[i][j][0] = 0;
        scl_cell_m1[i][j][1] = 0;
        }}

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double C_cell[2][2];
    double exp_m1[2];

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
          scl_cell_m1[i][j][0] += C_cell[i][j]*exp_m1[0];
          scl_cell_m1[i][j][1] += C_cell[i][j]*exp_m1[1];
          }}
    }

    // write to the global field memory
    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        i_dep = i_grid_glob + j + Nx_grid*i;

        scl_m0[i_dep] = scl_m0[i_dep] + scl_cell_m0[i][j];

        scl_m1[i_dep] = scl_m1[i_dep] + (double2) {scl_cell_m1[i][j][0],
                                                   scl_cell_m1[i][j][1]};
        }}
  }
}}
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
  __global double *vec_z_m0,
  __global double2 *vec_x_m1,
  __global double2 *vec_y_m1,
  __global double2 *vec_z_m1)
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

    // allocate few integer counters
    uint i,j,k,i_dep;

    // allocate privite cell array for the deposition
    double vec_cell_m0[3][2][2];
    double vec_cell_m1[3][2][2][2];

    for (k=0; k<3; k++){
      for (i=0; i<2; i++){
        for (j=0; j<2; j++){
          vec_cell_m0[k][i][j] = 0.;
          vec_cell_m1[k][i][j][0] = 0.;
          vec_cell_m1[k][i][j][1] = 0.;
        }}}

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double C_cell[2][2];
    double exp_m1[2];

    double xp, yp, zp, wp,rp,rp_inv,jp_proj;
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

      rp_inv = 0;
      if (rp>0){rp_inv = 1./rp;}

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

      for (k=0;k<3;k++){
        for (i=0;i<2;i++){
          for (j=0;j<2;j++){
            jp_proj = C_cell[i][j]*jp[k];
            vec_cell_m0[k][i][j] += C_cell[i][j]*jp[k];
            vec_cell_m1[k][i][j][0] += jp_proj*exp_m1[0];
            vec_cell_m1[k][i][j][1] += jp_proj*exp_m1[1];
          }}}
   }
    // write to the global field memory

    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        i_dep = i_grid_glob + j + Nx_grid*i;

        vec_x_m0[i_dep] = vec_x_m0[i_dep] + vec_cell_m0[0][i][j];
        vec_y_m0[i_dep] = vec_y_m0[i_dep] + vec_cell_m0[1][i][j];
        vec_z_m0[i_dep] = vec_z_m0[i_dep] + vec_cell_m0[2][i][j];

        vec_x_m1[i_dep] = vec_x_m1[i_dep] + (double2) {vec_cell_m1[0][i][j][0],
                                                       vec_cell_m1[0][i][j][1]};
        vec_y_m1[i_dep] = vec_y_m1[i_dep] + (double2) {vec_cell_m1[1][i][j][0],
                                                       vec_cell_m1[1][i][j][1]};
        vec_z_m1[i_dep] = vec_z_m1[i_dep] + (double2) {vec_cell_m1[2][i][j][0],
                                                       vec_cell_m1[2][i][j][1]};
        }}
  }
}}
}


// Gather linearly fields on the particles particle kernels
__kernel void project_vec6(
  __global uint *sorting_indx,
  __global double *ex_proj,
  __global double *ey_proj,
  __global double *ez_proj,
  __global double *bx_proj,
  __global double *by_proj,
  __global double *bz_proj,
  __global double *x,
  __global double *y,
  __global double *z,
  __global uint *indx_offset,
             uint Np,
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv,
  __constant uint *Nxm1Nrm1,
  __global double *ex_m0,
  __global double *ey_m0,
  __global double *ez_m0,
  __global double *bx_m0,
  __global double *by_m0,
  __global double *bz_m0,
  __global double2 *ex_m1,
  __global double2 *ey_m1,
  __global double2 *ez_m1,
  __global double2 *bx_m1,
  __global double2 *by_m1,
  __global double2 *bz_m1)
{
 // running kernels over the 4cells
 uint ip  = (uint) get_global_id(0);
 if (ip<Np)
  {
   uint ip_srtd = sorting_indx[ip];
   if (ip_srtd<Np)
   {
    // get cell number and period of grid
    int Nx_grid = (int) *Nx;
    int Nx_cell = Nx_grid - 1;
    int Nr_cell = (int) *Nr-1;

    double xmin_loc = *xmin;
    double rmin_loc = *rmin;

    double xp = x[ip_srtd];
    double yp = y[ip_srtd];
    double zp = z[ip_srtd];

    double rp = sqrt(yp*yp + zp*zp);

    double rp_inv = 1./rp;
    double dr_inv_loc = *dr_inv;
    double dx_inv_loc = *dx_inv;

    int ix = (int) floor( (xp-xmin_loc) * dx_inv_loc );
    int ir = (int) floor( (rp-rmin_loc) * dr_inv_loc );
    uint i_cell = ix + ir*Nx_cell;
    uint i_grid = ix + ir*Nx_grid;

    double sX1 = ( xp - xmin_loc )*dx_inv_loc - ix;
    double sX0 = 1.0 - sX1;
    double sR1 = ( rp - rmin_loc )*dr_inv_loc - ir;
    double sR0 = 1.0 - sR1;

    // allocate privitely some reused variables
    double C_cell[2][2];
    double exp_m1[2];

    exp_m1[0] = yp*rp_inv;
    exp_m1[1] = -zp*rp_inv;

    C_cell[0][0] = sR0*sX0;
    C_cell[0][1] = sR0*sX1;
    C_cell[1][0] = sR1*sX0;
    C_cell[1][1] = sR1*sX1;

    uint i,j,k,i_loc;

    // allocate privite cell array for the deposition
    // NB: multiplictaion of m1 by 2 accounts for Hermit symmetry
    double e_cell_m0[3][2][2] ;
    double b_cell_m0[3][2][2] ;
    double e_cell_m1[3][2][2][2] ;
    double b_cell_m1[3][2][2][2] ;

    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        i_loc = i_grid + j + Nx_grid*i;
        e_cell_m0[0][i][j] = ex_m0[i_loc];
        e_cell_m0[1][i][j] = ey_m0[i_loc];
        e_cell_m0[2][i][j] = ez_m0[i_loc];

        b_cell_m0[0][i][j] = bx_m0[i_loc];
        b_cell_m0[1][i][j] = by_m0[i_loc];
        b_cell_m0[2][i][j] = bz_m0[i_loc];

        e_cell_m1[0][i][j][0] = 2 * ((double) ex_m1[i_loc].s0);
        e_cell_m1[0][i][j][1] = 2 * ((double) ex_m1[i_loc].s1);

        e_cell_m1[1][i][j][0] = 2 * ((double) ey_m1[i_loc].s0);
        e_cell_m1[1][i][j][1] = 2 * ((double) ey_m1[i_loc].s1);

        e_cell_m1[2][i][j][0] = 2 * ((double) ez_m1[i_loc].s0);
        e_cell_m1[2][i][j][1] = 2 * ((double) ez_m1[i_loc].s1);

        b_cell_m1[0][i][j][0] = 2 * ((double) bx_m1[i_loc].s0);
        b_cell_m1[0][i][j][1] = 2 * ((double) bx_m1[i_loc].s1);

        b_cell_m1[1][i][j][0] = 2 * ((double) by_m1[i_loc].s0);
        b_cell_m1[1][i][j][1] = 2 * ((double) by_m1[i_loc].s1);

        b_cell_m1[2][i][j][0] = 2 * ((double) bz_m1[i_loc].s0);
        b_cell_m1[2][i][j][1] = 2 * ((double) bz_m1[i_loc].s1);
        }}


    double e_p[3], b_p[3];
    for (k=0;k<3;k++){
      e_p[k] = 0;
      b_p[k] = 0;
      }

    for (k=0;k<3;k++){
      for (i=0;i<2;i++){
        for (j=0;j<2;j++){
          e_p[k] += C_cell[i][j]*e_cell_m0[k][i][j];
          b_p[k] += C_cell[i][j]*b_cell_m0[k][i][j];

          e_p[k] += C_cell[i][j]*e_cell_m1[k][i][j][0]*exp_m1[0];
          e_p[k] -= C_cell[i][j]*e_cell_m1[k][i][j][1]*exp_m1[1];
          b_p[k] += C_cell[i][j]*b_cell_m1[k][i][j][0]*exp_m1[0];
          b_p[k] -= C_cell[i][j]*b_cell_m1[k][i][j][1]*exp_m1[1];
          }}}

      ex_proj[ip_srtd] += e_p[0] ;
      ey_proj[ip_srtd] += e_p[1] ;
      ez_proj[ip_srtd] += e_p[2] ;

      bx_proj[ip_srtd] += b_p[0] ;
      by_proj[ip_srtd] += b_p[1] ;
      bz_proj[ip_srtd] += b_p[2] ;
   }
  }
}
