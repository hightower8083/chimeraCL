import matplotlib.pyplot as plt
from tests.timings.time_align import run_test

num_ps = range(10**6,7*10**6,10**6)
Nint = 20
Nheatup = 5

answers_cpu = [0,]
answers_gpu = None

gpu_noal_np = []
cpu_noal_np = []

for nump in num_ps:
    gpu_noal_np.append(run_test(answers=answers_gpu,Np=nump,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))
    cpu_noal_np.append(run_test(answers=answers_cpu,Np=nump,aligned=False,
                                Nint=Nint,Nheatup=Nheatup))

print("Finished scan, making plot and saving it as 'scan_align.pdf'")
fig, ax_np = plt.subplots(1,1,figsize=(6,6))
cols = ['b','g',]

np_scans = [gpu_noal_np, cpu_noal_np]
scans_legend = ['GPU','CPU',]

for i in range(2):
    dat = np_scans[i]
    ax_np.semilogx(num_ps,dat,'o-',c=cols[i])

ax_np.legend(scans_legend)
ax_np.set_xlabel('Number of particles')
ax_np.set_ylabel('Timing (ms)')
ax_np.set_xlim(0,)
ax_np.set_ylim(0,)
plt.savefig('scan_align.pdf')
plt.show()
