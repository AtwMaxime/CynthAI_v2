import numpy as np
m = np.genfromtxt('checkpoints/cheater/metrics.csv', delimiter=',', names=True)
updates = m['update'][::20]
pools = m['pool'][::20]
print('update  pool')
for u, p in zip(updates, pools):
    print(f'{int(u):6d}  {int(p):3d}')
print(f'... {int(m["update"][-1]):6d}  {int(m["pool"][-1]):3d}')