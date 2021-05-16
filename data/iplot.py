import matplotlib.pyplot as plt
import numpy as np

meta = np.genfromtxt('meta.txt', delimiter=',')

sx = meta[0]; sy = meta[1]
x,y = np.meshgrid(np.arange(0, sx, 1), np.arange(0, sy, 1))

xn = x * 1 + y * (-0.5)
yn = x * 0 + y * (-0.8660254037844386)


px = np.genfromtxt('ipx.csv', delimiter=',')
py = np.genfromtxt('ipy.csv', delimiter=',')
ocpy = np.genfromtxt('ioccupancy.csv', delimiter=',')

npx = np.multiply(px, ocpy)
npy = np.multiply(py, ocpy)

fig, axs = plt.subplots(1, 1)

axs.quiver(xn, yn, npx, npy)
axs.axis('equal')
# axs.set_xlim(250,400)
plt.savefig('iplot.png')
plt.show()
