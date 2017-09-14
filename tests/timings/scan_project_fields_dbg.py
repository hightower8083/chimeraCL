import matplotlib.pyplot as plt
from timings.time_project_fields import run_test

Nint = 1
Nheatup = 0

cpu_noal_np = []

for i in range(1000):
    cpu_noal_np.append(run_test(answer=0,Np=2*10**6,dims=(1024,256),
                                aligned=False,
                                Nint=Nint,Nheatup=Nheatup))
    print(i)


print("Finished scan, making plot.")

fig, ax_np = plt.subplots(1,1,figsize=(6,6))
ax_np.plot(cpu_noal_np,'o-')

ax_np.set_ylabel('Timing (ms)')
ax_np.set_xlim(0,)
ax_np.set_ylim(0,)

plt.show()
