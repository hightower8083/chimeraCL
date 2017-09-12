import matplotlib.pyplot as plt
from timings.time_project_charge import run_test

Nint = 30
Nheatup = 10
num_ps = range(10**6,7*10**6,10**6)
grid_sizes = range(1,9,1)

answers_cpu = [0,]
answers_gpu = None

gpu_al_np = []
gpu_noal_np = []
cpu_noal_np = []
cpu_al_np = []

gpu_al_gr = []
gpu_noal_gr = []
cpu_noal_gr = []
cpu_al_gr = []

for nump in num_ps:
    gpu_noal_np.append(run_test(answers=answers_gpu,Np=nump,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))
    gpu_al_np.append(run_test(answers=answers_gpu,Np=nump,aligned=True,
                              Nint=Nint,Nheatup=Nheatup))
    cpu_noal_np.append(run_test(answers=answers_cpu,Np=nump,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))
    cpu_al_np.append(run_test(answers=answers_cpu,Np=nump,aligned=True,
                              Nint=Nint,Nheatup=Nheatup))

for grid_size in grid_sizes:
    dims = 256*grid_size, 64*grid_size
    nump = 2*10**6
    gpu_noal_gr.append(run_test(answers=answers_gpu,Np=nump,dims=dims,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))
    gpu_al_gr.append(run_test(answers=answers_gpu,Np=nump,dims=dims,aligned=True,
                              Nint=Nint,Nheatup=Nheatup))
    cpu_noal_gr.append(run_test(answers=answers_cpu,Np=nump,dims=dims,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))
    cpu_al_gr.append(run_test(answers=answers_cpu,Np=nump,dims=dims,aligned=True,
                              Nint=Nint,Nheatup=Nheatup))

print("Finished scan, making plot and saving it as 'scan_project_charge.pdf'")

fig, (ax_np,ax_gr) = plt.subplots(1,2,figsize=(12,6))
cols = ['b','g','r','y']

np_scans = [gpu_al_np,gpu_noal_np,cpu_noal_np,cpu_al_np]
gr_scans = [gpu_al_gr,gpu_noal_gr,cpu_noal_gr,cpu_al_gr]
scans_legend = ['GPU pre-aligned','GPU not-aligned',
                'CPU not-aligned','CPU pre-aligned']

for i in range(4):
    dat = np_scans[i]
    ax_np.semilogx(num_ps,dat,'o-',c=cols[i])

for i in range(4):
    dat = gr_scans[i]
    ax_gr.plot(grid_sizes,dat,'o-',c=cols[i])

ax_np.legend(scans_legend)
ax_np.set_xlabel('Number of particles')
ax_gr.set_xlabel('Grid size factor i*(256,64)')
ax_np.set_ylabel('Timing (ms)')

ax_np.set_xlim(0,)
ax_np.set_ylim(0,)
ax_gr.set_xlim(0,)
ax_gr.set_ylim(0,)
plt.savefig('scan_project_charge.pdf')

plt.show()
