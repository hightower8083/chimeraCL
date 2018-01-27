# CHIMERA(.CL)
## Project of a PSATD PIC code on heterogeneous architectures

[![CRAN](https://img.shields.io/cran/l/devtools.svg)](LICENSE)

by Igor A Andriyash (<igor.andriyash@gmail.com>)

CHIMERA(.CL) is a project of a relativistic electromagnetic particle-in-cell code, based on a quasi-cylindric pseudo-spectral analytical time domain (PSATD) Maxwell solver. More details on this method can be found in the original publications [<cite>[1]</cite>,<cite>[2]</cite>,<cite>[3]</cite>]. 

The project spins off from the original code [CHIMERA](https://github.com/hightower8083/chimera) and is aimed to be firstly a playground to learn (py)openCL approach to GPGPU programming.
The second goal of CHIMERA(.CL) is to design the code with maximum flexibility:
- code should be divided into (i) wrapper classes that are independent of computational part (e.g. particles.py and grid.py), and (ii) compiled methods (cf ./methods/) as much general as  possible.
- computational data should be presented with 'device' abstraction
- all data structures should be grouped into containers of dictionary type with comprehensive naming

Presenly developed methods are using [openCL](https://www.khronos.org/opencl) via [PyOpenCL](https://mathema.tician.de/software/pyopencl) API which also benefits from a few methods provided by [Reikna](http://reikna.publicfields.net). 

Code also includes the [tvtk](http://docs.enthought.com/mayavi/tvtk/README.html)-based script to convert the output gerenated by the code into the [VTK](https://www.vtk.org) containers.

Examples of the simulation scripts and [Paraview](https://www.paraview.org) state-file for visualising the VTK files can be found in `./examples/` folder.

System requirements
- Linux or MacOS (not tested on Windows)
- python distribution with numpy and scipy (matplotlib is also helpful)
- [openCL](https://www.khronos.org/opencl) driver (tested with version 1.2 )
- [PyOpenCL](https://mathema.tician.de/software/pyopencl)
- [Reikna](http://reikna.publicfields.net) library
- [pyFFTW](https://github.com/hgomersall/pyFFTW) (Reikna's FFT crashes on Apple CPUs, cause Apple are geniuses, so it's replaced for them)
- path to the code should be known to python (e.g. exported to PYTHONPATH)

\[[1]\] Igor A. Andriyash, Remi Lehe and Agustin Lifschitz, *Laser-plasma interactions with a Fourier-Bessel particle-in-cell method*, Physics of Plasmas **23**, 033110 
(2016)

\[[2]\] Remi Lehe, Manuel Kirchen, Igor A. Andriyash, Brendan B. Godfrey and Jean-Luc Vay, *A spectral, quasi-cylindrical and dispersion-free Particle-In-Cell algorithm*, 
Computer Physics Communications **203**, 66 (2016)

\[[3]\] Igor A. Andriyash, Remi Lehe and Victor Malka, *A spectral unaveraged algorithm for free electron laser simulations*, Journal of Computational Physics **282**, 397 (2015)

[1]:http://dx.doi.org/10.1063/1.4943281
[2]:http://dx.doi.org/10.1016/j.cpc.2016.02.007
[3]:http://dx.doi.org/10.1016/j.jcp.2014.11.026
