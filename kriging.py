import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
from matplotlib import cm

plot3D = True

if False: # data from Oysand site
	points = {
	# known points (xyz & layer definitions)
		"oysts25" : { "xyz" : [7022988.02, 562647.45, 3.39], "D": 21.900, "layers": [0,1.5,5.6,8.15,11,16.4,19.8,21.9] },
		"oysts47" :	{ "xyz" : [7022993.96, 562647.47, 3.34], "D": 21.850, "layers": [0,1.6,5.60,6.2,10.7,16.7,20.2,21.85] },
		"oysts29" :	{ "xyz" : [7022990.99, 562644.43, 3.28], "D": 21.825, "layers": [0,1.4,5.35,8.1,12.2,17.4,21.1,21.825] },
		"oysts29 " :{ "xyz" : [7022991.00, 562644.43, 3.28], "D": 21.825, "layers": [] }, # force a calculation with empty pos
		"oysts41" :	{ "xyz" : [7022987.96, 562641.44, 3.17], "D": 21.850, "layers": [0,1.4,4.95,9.4,12.4,18.1,21.1,21.85] },
		"oysts45" :	{ "xyz" : [7022993.99, 562641.47, 3.21], "D": 21.800, "layers": [0,1.4,5.2,9.8,12.4,18.4,21.6, 21.8] },
	# unknown points (xyz)
		"oysts24" : { "xyz" : [7022990.99, 562642.96, 3.25], "D": 21.825, "layers": [0, 1.5, 5.35, 9.0, 12.5, 17.5, 21.4, 21.825] },
		"oysts26" : { "xyz" : [7022992.49, 562644.48, 3.24], "D": 21.775, "layers": [0, 1.5, 5.6, 7.9, 12.1, 17.5, 21.1, 21.775] },
		"oysts27" : { "xyz" : [7022989.49, 562644.42, 3.31], "D": 21.825, "layers": [0, 1.45, 5.34, 8.1, 12.1, 17.5, 20.75, 21.825] },
		"oysts31" : { "xyz" : [7022990.92, 562647.44, 3.40], "D": 21.825, "layers": [0, 1.57, 6.4, 7.25, 11.2, 16.4, 20.1, 21.825] },
		"oysts35" : { "xyz" : [7022990.98, 562645.96, 3.38], "D": 21.825, "layers": [0, 1.52, 6.0, 7.4, 11.6, 16.9, 20.7, 21.825] },
		"oysts36" : { "xyz" : [7022987.99, 562644.46, 3.33], "D": 21.825, "layers": [0, 1.4, 5.3, 7.9, 11.85, 17.3, 20.5, 21.825] },
		"oysts39" : { "xyz" : [7022991.02, 562641.46, 3.26], "D": 21.850, "layers": [0, 1.25, 5.0, 10.4, 12.75, 18.5, 21.5, 21.850] },
		"oysts46" : { "xyz" : [7022994.00, 562644.44, 3.18], "D": 21.850, "layers": [0, 1.65, 5.6, 7.5, 11.65, 17.6, 21.2, 21.850] }
	}
else: # data from Kjellstad site
	points = {
		"kjts01" : { "xyz" : [6626339.01, 570614.14, 17.64], "D":  20.02, "layers": [ 0.1, 0.84, 2.31, 8.02, 8.76, 15.16, 17.26, 20.02] }, 
		"kjts01 " : { "xyz" : [6626339.011, 570614.14, 17.64], "D":  20.02, "layers": [] }, # force a calculation with empty pos
		"kjts02" : { "xyz" : [6626341.37, 570611.09, 17.71], "D":  20.02, "layers": [ 0.1, 1.03, 2.15, 7.97, 8.55, 14.85, 17.05, 20.02] },
		"kjts03" : { "xyz" : [6626342.59, 570616.45, 17.63], "D":  20.02, "layers": [ 0.1, 1.26, 2.52, 8.07, 8.80, 15.25, 17.33, 20.02] },
		"kjts04" : { "xyz" : [6626335.53, 570611.78, 17.67], "D":  20.02, "layers": [ 0.1, 0.82, 2.55, 7.60, 8.26, 15.07, 17.16, 20.02] },
		"kjts05" : { "xyz" : [6626336.67, 570617.26, 17.71], "D":  20.21, "layers": [ 0.1, 1.04, 2.49, 8.54, 9.31, 15.40, 17.61, 20.21] },
		"kjts06" : { "xyz" : [6626342.02, 570613.70, 17.67], "D":  20.03, "layers": [] },
		"kjts07" : { "xyz" : [6626340.46, 570613.93, 17.71], "D":  20.05, "layers": [] },
		"kjts08" : { "xyz" : [6626337.36, 570614.41, 17.66], "D":  20.02, "layers": [] },
		"kjts09" : { "xyz" : [6626336.10, 570614.50, 17.69], "D":  20.05, "layers": [] },
		"kjts10" : { "xyz" : [6626338.40, 570611.39, 17.69], "D":  20.03, "layers": [] },
		"kjts11" : { "xyz" : [6626338.75, 570612.91, 17.65], "D":  20.02, "layers": [] },
		"kjts12" : { "xyz" : [6626339.32, 570615.63, 17.72], "D":  20.74, "layers": [] },
		"kjts13" : { "xyz" : [6626339.55, 570616.87, 17.64], "D":  20.01, "layers": [] }
	}

# keep track of these for each position
no_layers = [] # number of layers
x_s = [] # coords
y_s = [] # coords
pos_name = []

for pos in points:
	no_layers.append(len(points[pos]["layers"]))
	x_s.append(points[pos]["xyz"][0])
	y_s.append(points[pos]["xyz"][1])
	pos_name.append(pos)

N_min = min(no_layers)
N_max = max(no_layers)

dxy = 0.2 # internal interpolation interval
Dxy = .5 # how far beyond points to calculate 
grid_x = np.arange( (min(x_s)-Dxy), (max(x_s)+Dxy) + dxy, dxy )
grid_y = np.arange( (min(y_s)-Dxy), (max(y_s)+Dxy) + dxy, dxy )
grid_z = [] # results we will calculate (list of np.arrays) 

while N_min<N_max:
# group known and unknown points
	knowns = []
	unknowns = []
	for num, length in enumerate(no_layers):
		if length > N_min:
			knowns.append(pos_name[num])
		else:
			unknowns.append(pos_name[num])	

# build set of knowns
	fixed_points = []
	for k in knowns:
		fixed_points.append([
			points[k]["xyz"][0],
			points[k]["xyz"][1],
			points[k]["xyz"][2] - points[k]["layers"][N_min]] # z-coord of layer
		)
	model_data = np.array(fixed_points)

	p_x = model_data[:, 0]
	p_y = model_data[:, 1]
	p_z = model_data[:, 2]

	if len(set(p_z)) <= 1: # interpolate constant value
# interpolate each point
		for uk in unknowns:
			points[uk]["layers"].append(
				max( points[uk]["xyz"][2] - p_z[0], 0 ) # depth from coord
				)
# interpolate grid
		grid_z_i = np.empty( (grid_y.shape[0],grid_x.shape[0]) )
		grid_z_i.fill(points[next(iter(points))]["xyz"][2] - p_z[0])

		grid_z.append(
			grid_z_i.view(np.ma.MaskedArray)
		)
	else: # use kriging
		OK = OrdinaryKriging(
			p_x, p_y, p_z,
			variogram_model="gaussian", # or: linear,power,spherical,exponential
			verbose=False,
			enable_plotting=False,
		)

# interpolate each point
		for uk in unknowns:
			x_i = points[uk]["xyz"][0]
			y_i = points[uk]["xyz"][1]
			z_i = points[uk]["xyz"][2]
			p_zi, ss_i = OK.execute( "grid", x_i, y_i )
			# depth from coord (min 0!)
			D_i = max( round(( z_i - p_zi.compressed()[0] )*100) / 100, 0 )

			points[uk]["layers"].append( D_i )

# interpolate grid
		zi, ss = OK.execute("grid", grid_x, grid_y)
		grid_z.append( zi )

	N_min += 1

# present results
if plot3D:
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X, Y = np.meshgrid(grid_x, grid_y)

	for g in grid_z: # layers
		surf = ax.plot_surface( X, Y, g, alpha=.5)#, cmap=cm.coolwarm, linewidth=0, antialiased=False )
	#fig.colorbar(surf, shrink=0.5, aspect=5)

# plot boreholes
	dz = .25
	for p in points:
		x_i = points[p]["xyz"][0]
		y_i = points[p]["xyz"][1]
		z_i = points[p]["xyz"][2]
		z_i_D = z_i -  points[p]["D"]

		ax.scatter3D( x_i, y_i, z_i+dz, color='black') # ball
		ax.plot([x_i,x_i], [y_i,y_i], [z_i,z_i_D], color='black')#, label='parametric curve') # line

		ax.text(x_i, y_i, z_i, p, None)

	ax.set_xlabel('Northing (m)')
	ax.set_ylabel('Easting (m)')
	ax.set_zlabel('Elevation (m)')
	plt.gca().invert_xaxis()
	plt.show()
		
for g in grid_z:
	print(g)

for p in points:
	print(p,points[p]["layers"])