import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


# data from study
rates_data = [ # could have been estimated for each unit
	[1/60 * 1000] * 13, # mm/sek
	[2/60 * 1000] * 13,
	[3/60 * 1000] * 13,
	[4/60 * 1000] * 13,
	[5/60 * 1000] * 13
]
forces_data = [
	[0.92, 1.27, 1.01, 0.62, 0.84, 7.16, 5.85, 7.92, 11.59, 11.84, 10.56, 18.80, 20.65], 
	[1.66, 1.91, 2.09, 2.31, 2.55, 8.06, 9.60, 9.85, 13.10, 17.05, 13.57, 22.86, 22.86], 
	[2.15, 2.62, 2.99, 3.25, 3.43, 8.54, 11.67, 12.52, 15.50, 17.14, 17.46, 25.51, 28.40], 
	[2.32, 2.67, 2.91, 3.83, 3.75, 8.49, 12.76, 13.46, 14.93, 20.24, 19.39, 24.75, 23.96], 
	[3.30, 3.82, 4.42, 5.64, 5.42, 10.56, 12.01, 13.92, 14.87, 17.81, 16.79, 28.91, 32.51]
]
factors_data = [
	[0.43, 0.48, 0.34, 0.19, 0.24, 0.84, 0.50, 0.63, 0.75, 0.69, 0.60, 0.74, 0.73], 
	[0.77, 0.73, 0.70, 0.71, 0.74, 0.94, 0.82, 0.79, 0.84, 0.99, 0.78, 0.90, 0.80], 
	[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], 
	[1.08, 1.02, 0.97, 1.18, 1.09, 0.99, 1.09, 1.07, 0.96, 1.18, 1.11, 0.97, 0.84], 
	[1.53, 1.46, 1.48, 1.74, 1.58, 1.24, 1.03, 1.11, 0.96, 1.04, 0.96, 1.13, 1.14]
]
label_data = ['Very slow (16.7 mm/s)', 'Slow (16.7 mm/s)', 'Normal (16.7 mm/s)', 'Fast (16.7 mm/s)', 'Very fast (16.7 mm/s)']


# surface calculations
n_xy = 500 # number of points

rate_min = 17
rate_max = 85
force_min = 0.001
force_max = 40

grid_rate = np.linspace(rate_min, rate_max, num=n_xy) # mm/seks
grid_force = np.linspace(force_min, force_max, num=n_xy) # kN
rate, force = np.meshgrid(grid_rate, grid_force)

# model parameters as function of rate
a = 8.02000000e-6 * rate**3 - 1.2001e-3 * rate**2 + 7.680927e-2 * rate -8.370417e-1
b = 3.2598e-4 * rate**2 - 2.167286e-2 * rate + 1.52108915
c = -1.01357e-3 * rate**2 + 1.2439644e-1 * rate + 2.0146433e-1
d = 2.81007567e-6 * rate**3 -4.85590747e-4 * rate**2 + 3.028811e-2 * rate + 3.487437e-1

# calculate model
k = d + ( a-d ) / ( 1 + ( force/c )**b )

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# plot model and datapoints
surf = ax.plot_wireframe(rate, force, k, rstride=30, cstride=30, color='#333', linewidth=0.5)
for r, f, k_val, l in zip(rates_data, forces_data, factors_data, label_data  ):
	ax.scatter3D( r, f, k_val, label=l)

ax.set_xlim(15, 90)
ax.set_ylim(0, 40)
ax.set_zlim(0, 2)

ax.set_xlabel('Rate (mm/s)', multialignment='center')
ax.set_ylabel('Measured push force (kN)', multialignment='center')
ax.set_zlabel('Correction factor, k (-)', multialignment='center')

plt.gca().invert_xaxis()
leg = plt.legend(loc='best', fancybox=True)

plt.show()