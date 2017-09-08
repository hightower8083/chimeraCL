// this is a source of grid kernels for chimeraCL project

// Depose particles "weights" onto 2D grid via linear projection,
// using groups of 4 cells and non-concurrent steps for each cell
// in a group by calling kernel with a different offset (0,1,2,3)
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
  __constant uint *Nx,
  __constant double *xmin,
  __constant double *dx_inv,
  __constant uint *Nr,
  __constant double *rmin,
  __constant double *dr_inv,
  __constant uint *Nxm1Nrm1_4,
  __global double *scl_m0)
{
  // running kernels over the 4cells
  uint i_cell = get_global_id(0);
  if (i_cell < *Nxm1Nrm1_4)
   {
    // get cell number and period of 4cell-grid
    uint Nx_grid = *Nx;
    uint Nx_cell = Nx_grid-1;
    uint Nx_2 = Nx_cell/2;

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
    double scl_cell_m0[2][2] = {{0.,0.},{0.,0.}};

    // allocate privitely some reused variables
    double sX0, sX1, sR0, sR1;
    double s00, s01, s10, s11;

    double xp, yp, zp, wp,rp;
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
      wp = w[ip_srtd];

      rp = sqrt(yp*yp + zp*zp);

      sX0 = ( xp - xmin_loc )*dx_inv_loc - ix;
      sX1 = 1.0 - sX0;
      sR0 = ( rp - rmin_loc )*dr_inv_loc - ir;
      sR1 = 1.0 - sR0;

      sX0 *= wp;
      sX1 *= wp;

      s00 = sX0*sR0;
      s01 = sX1*sR0;
      s10 = sX0*sR1;
      s11 = sX1*sR1;

      scl_cell_m0[0][0] += s00;
      scl_cell_m0[0][1] += s01;
      scl_cell_m0[1][0] += s10;
      scl_cell_m0[1][1] += s11;
     }

    // write to the global field memory
    scl_m0[i_grid_glob] += scl_cell_m0[0][0];
    scl_m0[i_grid_glob+1] += scl_cell_m0[0][1];
    scl_m0[i_grid_glob+Nx_grid] += scl_cell_m0[1][0];
    scl_m0[i_grid_glob+Nx_grid+1] += scl_cell_m0[1][1];
   }
}

__kernel void depose_vector(
    __global uint *cell_offset,
    __global double *x,
    __global double *y,
    __global double *z,
    __global double *ux,
    __global double *uy,
    __global double *uz,
    __global double *g_inv,
    __global double *w,
    __global uint *indx_offset,
    __constant uint *Nx,
    __constant double *xmin,
    __constant double *dx_inv,
    __constant uint *Nr,
    __constant double *rmin,
    __constant double *dr_inv,
    __constant uint *Nxm1Nrm1_4,
    __global double *vec_x_m0,
    __global double *vec_y_m0,
    __global double *vec_z_m0)
{
    // running kernels over the 4cells
    uint i_cell = get_global_id(0);
    if (i_cell < *Nxm1Nrm1_4)
    {
        // get cell number and period of 4cell-grid
        uint Nx_grid = *Nx;
        uint Nx_cell = Nx_grid-1;
        uint Nx_2 = Nx_cell/2;

        // get indicies of 4cell origin (left-bottom) on 4cell-grid
        uint ir = i_cell/Nx_2;
        uint ix = i_cell - ir*Nx_2;

        // convert 4cell indicies to global grid
        ix *= 2;
        ir *= 2;

        // apply offset whithin 4cell (0,1,2,3)
        //  _________
        // | 2  | 3  |
        // |____|____|
        // | 0  | 1  |
        // |____|____|
        //
        if (*cell_offset==1)
            {ix += 1;}
        else if (*cell_offset==2)
            {ir += 1;}
        else if (*cell_offset==3)
            {ix += 1;ir += 1;}

        // get 1D indicies of the selected
        // cell and grid node on the global grid
        uint i_cell_glob = ix + ir*Nx_cell;
        uint i_grid_glob = ix + ir*Nx_grid;

        // get particles indicies in the selected cell
        uint ip_start = indx_offset[i_cell_glob];
        uint ip_end = indx_offset[i_cell_glob+1];

        // allocate privite cell array for the deposition
        double vec_x_cell_m0[2][2] = {{0.,0.},{0.,0.}};
        double vec_y_cell_m0[2][2] = {{0.,0.},{0.,0.}};
        double vec_z_cell_m0[2][2] = {{0.,0.},{0.,0.}};

        // allocate privitely some reused variables
        double sX0, sX1, sR0, sR1;
        double s00, s01, s10, s11;
        double xp, yp, zp, rp, wp;
        double uxp, uyp, uzp;
        double rmin_loc = *rmin;
        double dr_inv_loc = *dr_inv;
        double xmin_loc = *xmin;
        double dx_inv_loc = *dx_inv;

        // run over the particles for linear deposition
        for (uint ip=ip_start; ip<ip_end; ip++)
        {
            xp = x[ip];
            yp = y[ip];
            zp = z[ip];
            uxp = ux[ip];
            uyp = uy[ip];
            uzp = uz[ip];
            wp = w[ip]*g_inv[ip];

            rp = sqrt(yp*yp + zp*zp);

            sX0 = ( xp - xmin_loc )*dx_inv_loc - ix;
            sX1 = 1.0 - sX0;
            sR0 = ( rp - rmin_loc )*dr_inv_loc - ir;
            sR1 = 1.0 - sR0;

            sX0 *= wp;
            sX1 *= wp;

            s00 = sX0*sR0;
            s01 = sX1*sR0;
            s10 = sX0*sR1;
            s11 = sX1*sR1;

            vec_x_cell_m0[0][0] += s00 * uxp;
            vec_x_cell_m0[0][1] += s01 * uxp;
            vec_x_cell_m0[1][0] += s10 * uxp;
            vec_x_cell_m0[1][1] += s11 * uxp;

            vec_y_cell_m0[0][0] += s00 * uyp;
            vec_y_cell_m0[0][1] += s01 * uyp;
            vec_y_cell_m0[1][0] += s10 * uyp;
            vec_y_cell_m0[1][1] += s11 * uyp;

            vec_z_cell_m0[0][0] += s00 * uzp;
            vec_z_cell_m0[0][1] += s01 * uzp;
            vec_z_cell_m0[1][0] += s10 * uzp;
            vec_z_cell_m0[1][1] += s11 * uzp;
        }

        // write to the global field memory
        vec_x_m0[i_grid_glob] += vec_x_cell_m0[0][0];
        vec_x_m0[i_grid_glob+1] += vec_x_cell_m0[0][1];
        vec_x_m0[i_grid_glob+Nx_grid] += vec_x_cell_m0[1][0];
        vec_x_m0[i_grid_glob+Nx_grid+1] += vec_x_cell_m0[1][1];

        vec_y_m0[i_grid_glob] += vec_y_cell_m0[0][0];
        vec_y_m0[i_grid_glob+1] += vec_y_cell_m0[0][1];
        vec_y_m0[i_grid_glob+Nx_grid] += vec_y_cell_m0[1][0];
        vec_y_m0[i_grid_glob+Nx_grid+1] += vec_y_cell_m0[1][1];

        vec_z_m0[i_grid_glob] += vec_z_cell_m0[0][0];
        vec_z_m0[i_grid_glob+1] += vec_z_cell_m0[0][1];
        vec_z_m0[i_grid_glob+Nx_grid] += vec_z_cell_m0[1][0];
        vec_z_m0[i_grid_glob+Nx_grid+1] += vec_z_cell_m0[1][1];
    }
}

__kernel void project_scalar(
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
    uint i_cell = get_global_id(0);
    if (i_cell < *Nxm1Nrm1)
    {
        // get cell number and grid period
        uint Nx_grid = *Nx;
        uint Nx_cell = Nx_grid-1;

        // get indicies of cell origin (left-bottom)
        uint ir = i_cell/Nx_cell;
        uint ix = i_cell - ir*Nx_cell;

        // get 1D indicies of the selected grid node
        uint i_grid = ix + ir*Nx_grid;

        // get particles indicies in the selected cell
        uint ip_start = indx_offset[i_cell] ;
        uint ip_end = indx_offset[i_cell+1] ;

        // allocate privite cell array for the deposition
        double scl_cell_m0[2][2] ;

        scl_cell_m0[0][0] = scl_m0[i_grid] ;
        scl_cell_m0[0][1] = scl_m0[i_grid+1] ;
        scl_cell_m0[1][0] = scl_m0[i_grid+Nx_grid] ;
        scl_cell_m0[1][1] = scl_m0[i_grid+Nx_grid+1] ;

        // allocate privitely some reused variables
        double sX0, sX1, sR0, sR1;
        double s00, s01, s10, s11;
        double xp, yp, zp, rp, scl_proj_p;
        double rmin_loc = *rmin;
        double dr_inv_loc = *dr_inv;
        double xmin_loc = *xmin;
        double dx_inv_loc = *dx_inv;

        // run over the particles for linear deposition
        for (uint ip=ip_start; ip<ip_end; ip++)
        {
            xp = x[ip];
            yp = y[ip];
            zp = z[ip];
            rp = sqrt(yp*yp + zp*zp);
            scl_proj_p = 0;

            sX0 = ( xp - xmin_loc )*dx_inv_loc - ix;
            sX1 = 1.0 - sX0;
            sR0 = ( rp - rmin_loc )*dr_inv_loc - ir;
            sR1 = 1.0 - sR0;

            s00 = sX0*sR0;
            s01 = sX1*sR0;
            s10 = sX0*sR1;
            s11 = sX1*sR1;

            scl_proj_p += s00*scl_cell_m0[0][0];
            scl_proj_p += s01*scl_cell_m0[0][1];
            scl_proj_p += s10*scl_cell_m0[1][0];
            scl_proj_p += s11*scl_cell_m0[1][1];

            scl_proj[ip] += scl_proj_p ;
        }
    }
}

__kernel void project_vec6(
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
    uint i_cell = get_global_id(0);
    if (i_cell < *Nxm1Nrm1)
    {
        // get cell number and grid period
        uint Nx_grid = *Nx;
        uint Nx_cell = Nx_grid-1;

        // get indicies of cell origin (left-bottom)
        uint ir = i_cell/Nx_cell;
        uint ix = i_cell - ir*Nx_cell;

        // get 1D indicies of the selected grid node
        uint i_grid = ix + ir*Nx_grid;

        // get particles indicies in the selected cell
        uint ip_start = indx_offset[i_cell] ;
        uint ip_end = indx_offset[i_cell+1] ;

        // allocate privite cell array for the deposition
        double vec_x_1_cell_m0[2][2] ;
        double vec_y_1_cell_m0[2][2] ;
        double vec_z_1_cell_m0[2][2] ;

        double vec_x_2_cell_m0[2][2] ;
        double vec_y_2_cell_m0[2][2] ;
        double vec_z_2_cell_m0[2][2] ;

        vec_x_1_cell_m0[0][0] = vec_x_1_m0[i_grid] ;
        vec_x_1_cell_m0[0][1] = vec_x_1_m0[i_grid+1] ;
        vec_x_1_cell_m0[1][0] = vec_x_1_m0[i_grid+Nx_grid] ;
        vec_x_1_cell_m0[1][1] = vec_x_1_m0[i_grid+Nx_grid+1] ;

        vec_y_1_cell_m0[0][0] = vec_y_1_m0[i_grid] ;
        vec_y_1_cell_m0[0][1] = vec_y_1_m0[i_grid+1] ;
        vec_y_1_cell_m0[1][0] = vec_y_1_m0[i_grid+Nx_grid] ;
        vec_y_1_cell_m0[1][1] = vec_y_1_m0[i_grid+Nx_grid+1] ;

        vec_z_1_cell_m0[0][0] = vec_z_1_m0[i_grid] ;
        vec_z_1_cell_m0[0][1] = vec_z_1_m0[i_grid+1] ;
        vec_z_1_cell_m0[1][0] = vec_z_1_m0[i_grid+Nx_grid] ;
        vec_z_1_cell_m0[1][1] = vec_z_1_m0[i_grid+Nx_grid+1] ;

        vec_x_2_cell_m0[0][0] = vec_x_2_m0[i_grid] ;
        vec_x_2_cell_m0[0][1] = vec_x_2_m0[i_grid+1] ;
        vec_x_2_cell_m0[1][0] = vec_x_2_m0[i_grid+Nx_grid] ;
        vec_x_2_cell_m0[1][1] = vec_x_2_m0[i_grid+Nx_grid+1] ;

        vec_y_2_cell_m0[0][0] = vec_y_2_m0[i_grid] ;
        vec_y_2_cell_m0[0][1] = vec_y_2_m0[i_grid+1] ;
        vec_y_2_cell_m0[1][0] = vec_y_2_m0[i_grid+Nx_grid] ;
        vec_y_2_cell_m0[1][1] = vec_y_2_m0[i_grid+Nx_grid+1] ;

        vec_z_2_cell_m0[0][0] = vec_z_2_m0[i_grid] ;
        vec_z_2_cell_m0[0][1] = vec_z_2_m0[i_grid+1] ;
        vec_z_2_cell_m0[1][0] = vec_z_2_m0[i_grid+Nx_grid] ;
        vec_z_2_cell_m0[1][1] = vec_z_2_m0[i_grid+Nx_grid+1] ;

        // allocate privitely some reused variables
        double sX0, sX1, sR0, sR1;
        double s00, s01, s10, s11;
        double xp, yp, zp, rp;
        double vec_x_1_proj_p, vec_y_1_proj_p, vec_z_1_proj_p;
        double vec_x_2_proj_p, vec_y_2_proj_p, vec_z_2_proj_p;
        double rmin_loc = *rmin;
        double dr_inv_loc = *dr_inv;
        double xmin_loc = *xmin;
        double dx_inv_loc = *dx_inv;

        // run over the particles for linear deposition
        for (uint ip=ip_start; ip<ip_end; ip++)
        {
            xp = x[ip];
            yp = y[ip];
            zp = z[ip];
            rp = sqrt(yp*yp + zp*zp);

            vec_x_1_proj_p = 0;
            vec_y_1_proj_p = 0;
            vec_z_1_proj_p = 0;

            vec_x_2_proj_p = 0;
            vec_y_2_proj_p = 0;
            vec_z_2_proj_p = 0;

            sX0 = ( xp - xmin_loc )*dx_inv_loc - ix;
            sX1 = 1.0 - sX0;
            sR0 = ( rp - rmin_loc )*dr_inv_loc - ir;
            sR1 = 1.0 - sR0;

            s00 = sX0*sR0;
            s01 = sX1*sR0;
            s10 = sX0*sR1;
            s11 = sX1*sR1;

            vec_x_1_proj_p += s00*vec_x_1_cell_m0[0][0];
            vec_x_1_proj_p += s01*vec_x_1_cell_m0[0][1];
            vec_x_1_proj_p += s10*vec_x_1_cell_m0[1][0];
            vec_x_1_proj_p += s11*vec_x_1_cell_m0[1][1];

            vec_y_1_proj_p += s00*vec_y_1_cell_m0[0][0];
            vec_y_1_proj_p += s01*vec_y_1_cell_m0[0][1];
            vec_y_1_proj_p += s10*vec_y_1_cell_m0[1][0];
            vec_y_1_proj_p += s11*vec_y_1_cell_m0[1][1];

            vec_z_1_proj_p += s00*vec_z_1_cell_m0[0][0];
            vec_z_1_proj_p += s01*vec_z_1_cell_m0[0][1];
            vec_z_1_proj_p += s10*vec_z_1_cell_m0[1][0];
            vec_z_1_proj_p += s11*vec_z_1_cell_m0[1][1];

            vec_x_2_proj_p += s00*vec_x_2_cell_m0[0][0];
            vec_x_2_proj_p += s01*vec_x_2_cell_m0[0][1];
            vec_x_2_proj_p += s10*vec_x_2_cell_m0[1][0];
            vec_x_2_proj_p += s11*vec_x_2_cell_m0[1][1];

            vec_y_2_proj_p += s00*vec_y_2_cell_m0[0][0];
            vec_y_2_proj_p += s01*vec_y_2_cell_m0[0][1];
            vec_y_2_proj_p += s10*vec_y_2_cell_m0[1][0];
            vec_y_2_proj_p += s11*vec_y_2_cell_m0[1][1];

            vec_z_2_proj_p += s00*vec_z_2_cell_m0[0][0];
            vec_z_2_proj_p += s01*vec_z_2_cell_m0[0][1];
            vec_z_2_proj_p += s10*vec_z_2_cell_m0[1][0];
            vec_z_2_proj_p += s11*vec_z_2_cell_m0[1][1];

            vec_x_1_proj[ip] += vec_x_1_proj_p ;
            vec_y_1_proj[ip] += vec_y_1_proj_p ;
            vec_z_1_proj[ip] += vec_z_1_proj_p ;

            vec_x_2_proj[ip] += vec_x_2_proj_p ;
            vec_y_2_proj[ip] += vec_y_2_proj_p ;
            vec_z_2_proj[ip] += vec_z_2_proj_p ;
        }
    }
}
