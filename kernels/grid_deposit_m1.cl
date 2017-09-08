// this is a source of grid kernels for chimeraCL project

#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

__kernel void depose_scalar(
    __global uint *cell_offset,
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
    __global double *rho_m0,
    __global cdouble_t *rho_m1)
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
        double rho_cell_m0[2][2] = {{0.,0.},{0.,0.}};
        double rho_cell_m1_re[2][2] = {{0.,0.},{0.,0.}};
        double rho_cell_m1_im[2][2] = {{0.,0.},{0.,0.}};

        // allocate privitely some reused variables
        double r, sX0, sX1, sR0, sR1;
        double s00,s01,s10,s11,yp,zp,r_inv,O1_re,O1_im;
        double rmin_loc = *rmin;
        double dr_inv_loc = *dr_inv;
        double xmin_loc = *xmin;
        double dx_inv_loc = *dx_inv;

        // run over the particles for linear deposition
        for (uint ip=ip_start; ip<ip_end; ip++)
        {
            yp = y[ip];
            zp = z[ip];
            r = sqrt(y[ip]*y[ip]+z[ip]*z[ip]);
            r_inv = 1./r;

            O1_re = yp*r_inv;
            O1_im = zp*r_inv;

            sX0 = ( x[ip] - xmin_loc )*dx_inv_loc - ix;
            sX1 = 1.0 - sX0;
            sR0 = ( r - rmin_loc )*dr_inv_loc - ir;
            sR1 = 1.0 - sR0;

            sX0 *= w[ip];
            sX1 *= w[ip];

            s00 = sX0*sR0;
            s10 = sX1*sR0;
            s01 = sX0*sR1;
            s11 = sX1*sR1;

            rho_cell_m0[0][0] += s00;
            rho_cell_m0[0][1] += s10;
            rho_cell_m0[1][0] += s01;
            rho_cell_m0[1][1] += s11;

            rho_cell_m1_re[0][0] += s00*O1_re;
            rho_cell_m1_re[0][1] += s10*O1_re;
            rho_cell_m1_re[1][0] += s01*O1_re;
            rho_cell_m1_re[1][1] += s11*O1_re;

            rho_cell_m1_im[0][0] += s00*O1_im;
            rho_cell_m1_im[0][1] += s10*O1_im;
            rho_cell_m1_im[1][0] += s01*O1_im;
            rho_cell_m1_im[1][1] += s11*O1_im;
        }

        // write to the global field memory

        rho_m0[i_grid_glob] += rho_cell_m0[0][0];
        rho_m0[i_grid_glob+1] += rho_cell_m0[0][1];
        rho_m0[i_grid_glob+Nx_grid] += rho_cell_m0[1][0];
        rho_m0[i_grid_glob+Nx_grid+1] += rho_cell_m0[1][1];

        rho_m1[i_grid_glob].real += rho_cell_m1_re[0][0];
        rho_m1[i_grid_glob+1].real += rho_cell_m1_re[0][1];
        rho_m1[i_grid_glob+Nx_grid].real += rho_cell_m1_re[1][0];
        rho_m1[i_grid_glob+Nx_grid+1].real += rho_cell_m1_re[1][1];

        rho_m1[i_grid_glob].imag += rho_cell_m1_im[0][0];
        rho_m1[i_grid_glob+1].imag += rho_cell_m1_im[0][1];
        rho_m1[i_grid_glob+Nx_grid].imag += rho_cell_m1_im[1][0];
        rho_m1[i_grid_glob+Nx_grid+1].imag += rho_cell_m1_im[1][1];
    }
}
