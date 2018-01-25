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

if (ix>0 && ix<Nx_cell-1 && ir<Nr_cell-1 && ir>=0 ){

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

if (ix>0 && ix<Nx_cell-1 && ir<Nr_cell-1 && ir>=0 ){
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
__kernel void gather_and_push(
  __global double *x,
  __global double *y,
  __global double *z,
  __global double *px,
  __global double *py,
  __global double *pz,
  __global double *g_inv,
  __global uint *sorting_indx,
  __global uint *indx_offset,
  __constant double *dt,
             uint Np,
             uint Np_stay,
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
  __global double *bz_m0)
{
 // running kernels over the 4cells
 uint ip  = (uint) get_global_id(0);
 if (ip<Np)
  {
   uint ip_srtd = sorting_indx[ip];
   if (ip_srtd<Np_stay)
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

    double u_p[3] = {px[ip_srtd], py[ip_srtd], pz[ip_srtd]};

    double dt_2 = 0.5*(*dt);

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

    C_cell[0][0] = sR0*sX0;
    C_cell[0][1] = sR0*sX1;
    C_cell[1][0] = sR1*sX0;
    C_cell[1][1] = sR1*sX1;

    uint i,j,k,i_loc;

    // allocate privite cell array for the deposition
    // NB: multiplictaion of m1 by 2 accounts for Hermit symmetry
    double e_cell_m0[3][2][2] ;
    double b_cell_m0[3][2][2] ;

    for (i=0;i<2;i++){
      for (j=0;j<2;j++){
        i_loc = i_grid + j + Nx_grid*i;
        e_cell_m0[0][i][j] = ex_m0[i_loc];
        e_cell_m0[1][i][j] = ey_m0[i_loc];
        e_cell_m0[2][i][j] = ez_m0[i_loc];

        b_cell_m0[0][i][j] = bx_m0[i_loc];
        b_cell_m0[1][i][j] = by_m0[i_loc];
        b_cell_m0[2][i][j] = bz_m0[i_loc];
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
          }}}

    double um[3], up[3], u0[3], t[3], s[3];

    for(k=0;k<3;k++){
      um[k] = u_p[k] + dt_2*e_p[k];
    }

    double g_p_inv = 1. / sqrt(1. + um[0]*um[0] + um[1]*um[1] + um[2]*um[2]);

    for(k=0;k<3;k++){
      t[k] = dt_2 * b_p[k] * g_p_inv;
      }

    double t2p1_m1_05 = 2. / (1. + t[0]*t[0] + t[1]*t[1] + t[2]*t[2]) ;

    for(k=0;k<3;k++){
      s[k] = t[k] * t2p1_m1_05;
      }

    u0[0] = um[0] + um[1]*t[2] - um[2]*t[1];
    u0[1] = um[1] - um[0]*t[2] + um[2]*t[0];
    u0[2] = um[2] + um[0]*t[1] - um[1]*t[0];

    up[0] = um[0] +  u0[1]*s[2] - u0[2]*s[1];
    up[1] = um[1] -  u0[0]*s[2] + u0[2]*s[0];
    up[2] = um[2] +  u0[0]*s[1] - u0[1]*s[0];

    for(int k=0;k<3;k++) {
      u_p[k] = up[k] + dt_2*e_p[k];
      }

    g_p_inv = 1. / sqrt(1. + u_p[0]*u_p[0] + u_p[1]*u_p[1] + u_p[2]*u_p[2]);

    px[ip_srtd] = u_p[0];
    py[ip_srtd] = u_p[1];
    pz[ip_srtd] = u_p[2];
    g_inv[ip_srtd] = g_p_inv;
   }
  }
}
