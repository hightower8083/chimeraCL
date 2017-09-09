import matplotlib.pyplot as plt
from timings.time_transform import run_test

Nint = 10
Nheatup = 30
grid_sizes = range(1,11,1)

gpu_noal_gr = []
cpu_noal_gr = []

for grid_size in grid_sizes:
    dims = 256*grid_size, 64*grid_size
    nump = 2*10**6
    gpu_noal_gr.append(run_test(answer=2,Np=nump,dims=dims,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))
    cpu_noal_gr.append(run_test(answer=0,Np=nump,dims=dims,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))

print("Finished scan, making plot and saving it as 'scan_transform.pdf'")

fig, ax_gr = plt.subplots(1,1,figsize=(6,6))
cols = ['b','g',]

gr_scans = [gpu_noal_gr,cpu_noal_gr,]
scans_legend = ['GPU', 'CPU']

for i in range(2):
    dat = gr_scans[i]
    ax_gr.plot(grid_sizes,dat,'o-',c=cols[i])

ax_gr.legend(scans_legend)
ax_gr.set_xlabel('Grid size factor i*(256,64)')
ax_gr.set_ylabel('Timing (ms)')
ax_gr.set_xlim(0,)
ax_gr.set_ylim(0,)
plt.savefig('scan_transform.pdf')

plt.show()
