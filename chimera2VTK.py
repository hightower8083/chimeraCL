"""
    Script to convert output of chimeraCL simulation to VTK files.
    VTK files can be plotted by any appropriate viewer, I use Paraview.
    Preferrably should be ran from the simulation folder, e.g.
    where ./diags/ folder is located. Script creates the folder VTK
    where is stores the resuts.

    USAGE:
    The only user parameter is Nt which defines number of asimuthal planes
    where fields should be constructed, e.g. mininal reasonable value is
    Nt=5, and for the nice 3D plots Nt>25 is recommended.

    Script take the path-to-data argument, and in chimeraCL it is
    by default ./diags/, e.g.
        python ~/CODES/chimeraCL/chimera2VTK.py diags/

    Additional argument 'latest' forces to treat only the last output, e.g.
        python ~/CODES/chimeraCL/chimera2VTK.py diags/ latest

    Script can be used in multi-proc mode via MPI, e.g.
        mpirun -np 4 python ~/CODES/chimeraCL/chimera2VTK.py diags/

    NB: If simulation was done in a frame moving at v_f and
        "frame" argument was given to the diagnostics,  the X-axis
        (longitudinal) will be defined as x' = x - v_f*t.

    Dependancies of the script:
        tvtk : for now (unfortunately) available only in Python2
        numba: to accelerate treatment (can be dropped if needed)
        h5py: to read chimeraCL output
        mpi4py: to run in multi-proc mode (can be dropped if needed)
"""

import numpy as np
from tvtk.api import tvtk, write_data
import h5py, sys, os
from numba import jit
from mpi4py.MPI import COMM_WORLD as comm

path = sys.argv[1]
if len(sys.argv)>2:
    selection = sys.argv[2]
else:
    selection = None

Nt = 25
Theta = 2*np.pi / (Nt-1) * np.arange(Nt)

@jit
def make_grid(Rgrid, Theta, Xgrid):
    y_plane = (np.cos(Theta)*Rgrid[:,None]).ravel()
    z_plane = (np.sin(Theta)*Rgrid[:,None]).ravel()
    points = np.empty([len(y_plane)*len(Xgrid),3])

    start = 0
    for ix in range(len(Xgrid)):
        end = start + len(y_plane)

        x_plane = Xgrid[ix]
        points[start:end,0] = x_plane
        points[start:end,1] = y_plane
        points[start:end,2] = z_plane
        start = end
    return points


@jit
def make_scalar(Nx, Nr, Nt, fld_stack):
    vals = np.empty(Nt*Nr*Nx)
    start = 0
    exps = []
    for m in range(len(fld_stack)):
        if m==0:
            exps.append(0.5*np.ones_like(Theta))
        else:
            exps.append(np.exp(1.j*m*Theta))

    for ix in range(Nx):
        end = start + Nt*Nr
        slice = np.zeros((Nr, Nt))
        for m in range(len(fld_stack)):
            slice += 2 * np.real(fld_stack[m][:,ix][:,None] * exps[m])

        vals[start:end] = slice.ravel()
        start = end
    return vals

def make_vtk_grid(Nx, Nr, Nt, pts):
    sgrid = tvtk.StructuredGrid(dimensions=(Nt, Nr, Nx))
    sgrid.points = pts
    return sgrid

def add_vtk_scalar(sgrid, scl, scl_name):
    indx = sgrid.point_data.add_array(scl.astype(np.float32))
    sgrid.point_data.get_array(indx).name = scl_name
    return sgrid

if comm.rank==0:
    if os.path.exists('vtk'):
        for fl in os.listdir('vtk'):
            os.remove('./vtk/'+fl)
    else:
        os.mkdir('./vtk/')

recs = os.listdir(path)
recs.sort()

if np.mod(len(recs)-1, comm.size) !=0:
    if comm.rank==0:
        print('Number of records is not dividable by number of procs ')

recs = recs[comm.rank::comm.size]
if selection=='latest':
    recs = [recs[-1],]

comm.barrier()

for rec in recs:
    record = h5py.File(path+rec,'r')

    base_str = '/data'
    flds_str = '/fields/'
    parts_str = '/species/'
    info_str = '/info/'

    Args = {}
    iter = int(rec.split('.')[0])
    iter_str = str(iter)

    for key in ['Xgrid', 'Rgrid', 'dx', 'M', 'dt', 'FrameVelocity']:
        Args[key] = record[base_str+info_str+key].value

    x0 = iter * Args['dt'] * Args['FrameVelocity']

    ## Adding Fields
    Rgrid = Args['Rgrid'] * (Args['Rgrid']>0)
    Xgrid = Args['Xgrid'] - x0
    Nx, Nr = len(Xgrid), len(Rgrid)

    if 'fields' in record[base_str].keys():
        rec_name_vtk = 'fields_' + iter_str
        pts = make_grid(Rgrid, Theta, Xgrid)
        gr = make_vtk_grid(Nx, Nr, Nt, pts)
        fld_list = list(record[base_str + flds_str].keys())

        for scl in fld_list:
            scl_name = str(scl)
            scl_data = record[base_str + flds_str + scl_name].value

            scl_m_re = scl_data[0]
            scl_m_re[0] = scl_m_re[1]   # axis treat
            fld_stack = [scl_m_re,]
            for m in range(1, scl_data.shape[0], 2):
                scl_m_re = scl_data[m]
                scl_m_im = scl_data[m+1]
                scl_m_re[0] = 0         # axis treat
                scl_m_im[0] = 0         # axis treat
                fld_stack.append(scl_m_re + 1.j*scl_m_im)

            vals = make_scalar(Nx, Nr, Nt, fld_stack)
            add_vtk_scalar(gr, vals, scl_name)

        write_data(gr, './vtk/' + rec_name_vtk)

    if 'species' in record[base_str].keys():
        spcs_list = record[base_str+parts_str].keys()

        for spcs in spcs_list:
            rec_name_vtk = str(spcs) + '_' + iter_str
            x = record[base_str + parts_str+spcs+'/x'].value
            y = record[base_str + parts_str+spcs+'/y'].value
            z = record[base_str + parts_str+spcs+'/z'].value
            x -= x0
            spc_vtk = tvtk.PolyData(points=np.vstack((x,y,z)).T)

            for comp in record[base_str + parts_str+spcs].keys():
                if str(comp) in ['x', 'y', 'z']:
                    continue
                comp_path = base_str + parts_str + spcs + '/' + comp
                comp_vals = record[comp_path].value
                indx = spc_vtk.point_data.add_array(comp_vals.astype(np.float32))
                spc_vtk.point_data.get_array(indx).name = comp

            write_data(spc_vtk, './vtk/' + rec_name_vtk)

    record.close()
    print('done '+rec)
