import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

meta = np.genfromtxt('meta.txt', delimiter=',')

sx = meta[0]; sy = meta[1]
x,y = np.meshgrid(np.arange(0, sx, 1), np.arange(0, sy, 1))

xn = x * 1 + y * (-0.5)
yn = x * 0 + y * (-0.8660254037844386)


px = np.genfromtxt('px.csv', delimiter=',')
py = np.genfromtxt('py.csv', delimiter=',')
ocpy = np.genfromtxt('occupancy.csv', delimiter=',')

print("Average occupancy: ", np.sum(ocpy)/(sx*sy))

npx = np.multiply(px, ocpy)
npy = np.multiply(py, ocpy)

fig, axs = plt.subplots(1, 1)

cx = sx/2; cy = sy/2
ncx = cx * 1 + cy * (-0.5)
ncy = cx * 0 + cy * (-0.8660254037844386)

axs.quiver(xn, yn, npx, npy)
# axs.add_patch(Rectangle((ncx-7/2-2, ncy-8/2),
#                         7, 8,
#                         fc ='none', 
#                         ec ='g',
#                         lw = 2))
axs.axis('equal')
# axs.set_xlim(250,400)
plt.show()
