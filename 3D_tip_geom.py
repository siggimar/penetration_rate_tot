import matplotlib.pyplot as plt
import numpy as np
import math

# tip geometry
dia = 0.057 # m
r = dia/2

offset = 3 * dia

pen_rate_labels = ['1 m/min','2 m/min','3 m/min','4 m/min','5 m/min']
pen_rates = [1/60 , 2/60 , 3/60 , 4/60 , 5/60] # m/seks
rpm = 25 * (2*math.pi) / 60 # rad/secs

X = []
Y = []
Z = []

k=0
for rate in pen_rates:
    t = np.linspace(0, 1/rate, num=300) # secs
    theta = t * rpm
    X.append( r*np.sin(theta) + k*offset )
    Y.append( r*np.cos(theta) )
    Z.append( t*rate )

# needed to set 3D aspect ratio
    if k == 0:
        xs = X[-1].tolist()
        ys = Y[-1].tolist()
        zs = Z[-1].tolist()
    else:
        xs += X[-1].tolist()
        ys += Y[-1].tolist()
        zs += Z[-1].tolist()
    k += 1

# ... aspect ratio
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
mid_x = (xs.max()+xs.min()) * 0.5
mid_y = (ys.max()+ys.min()) * 0.5
mid_z = (zs.max()+zs.min()) * 0.5

fig = plt.figure()
ax = fig.gca(projection='3d')

# plot each curve
for rate, x, y, z in zip( pen_rate_labels, X, Y, Z ):
    ax.plot(x, y, z, label=rate)
leg = plt.legend(loc='best', fancybox=True)

# ...aspect ratio by setting limits
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.invert_zaxis()
ax.set_ylabel('Distance y (m)', multialignment='center')
ax.set_xlabel('Distance x (m)', multialignment='center')
ax.set_zlabel('Depth (m)', multialignment='center')
plt.show()