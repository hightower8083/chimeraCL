import numpy as np
from tvtk.api import tvtk, write_data
import h5py, sys, os
from numba import jit

path = sys.argv[-1]
Nt = 48
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
        points[start:end,0] = y_plane
        points[start:end,1] = z_plane
        points[start:end,2] = x_plane
        start = end
    return points


@jit
def make_scalar(Nx, Nr, Nt, fld_stack):
    vals = np.empty(Nt*Nr*Nx)
    start = 0
    for ix in range(Nx):
        end = start + Nt*Nr
        slice = np.zeros((Nr, Nt))
        for m in range( (len(fld_stack)+1) / 2) :
            mode = fld_stack[m][:,ix][:,None]
            if m==0:
                mode *= 0.5

            slice += 2 * np.real( mode * np.exp(1.j*m*Theta) )

        vals[start:end] = slice.ravel()
        start = end
    return vals

def make_vtk_grid(Nx, Nr, Nt, pts):
    sgrid = tvtk.StructuredGrid(dimensions=(Nt, Nr, Nx))
    sgrid.points = pts
    return sgrid


def set_vtk_scalar(sgrid, scl, scl_name):
    sgrid.point_data.scalars = scl
    sgrid.point_data.scalars.name = scl_name
    return sgrid

def add_vtk_scalar(sgrid, scl, scl_name, indx):
    sgrid.point_data.scalars.add_array(scl)
    sgrid.point_data.scalars.get_array(indx).name = scl_name
    return sgrid

os.mkdir('./vtk/')

recs = os.listdir(path)
recs.sort()

for rec in recs:
    record = h5py.File(path+rec,'r')

    base_str = '/data'
    flds_str = '/fields/'
    parts_str = '/species/'
    info_str = '/info/'

    Args = {}
    iter = int(rec.split('.')[0])
    scl_list = list(record[base_str + flds_str].keys())

    for key in ['Xgrid', 'Rgrid', 'dx', 'M', 'dt']:
        Args[key] = record[base_str+info_str+key].value

    Rgrid = Args['Rgrid'][1:]
    Xgrid = Args['Xgrid']-iter*Args['dt']
    Nx, Nr = len(Xgrid), len(Rgrid)

    pts = make_grid(Rgrid, Theta, Xgrid-iter*Args['dt'])
    gr = make_vtk_grid(Nx, Nr, Nt, pts)


    scl_name = str(scl_list[0])
    scl_data = record[base_str + flds_str + scl_name].value
    fld_stack = [scl_data[0],]
    for m in range(1, Args['M']+1,2):
        fld_stack.append(scl_data[m] + 1.j*scl_data[m+1])

    vals = make_scalar(Nx, Nr, Nt, fld_stack)
    set_vtk_scalar(gr, vals, scl_name)

    indx = 1
    for scl in scl_list[1:]:
        scl_name = str(scl)
        scl_data = record[base_str + flds_str + scl_name].value
        fld_stack = [scl_data[0],]
        for m in range(1, Args['M']+1,2):
            fld_stack.append(scl_data[m] + 1.j*scl_data[m+1])

        vals = make_scalar(Nx, Nr, Nt, fld_stack)
        add_vtk_scalar(gr, vals, scl_name, indx)
        indx += 1

    rec_name_vtk = 'foo_' + str(int(rec.split('.')[0]))
    write_data(gr, './vtk/' + rec_name_vtk)
    print('done '+rec_name_vtk)
