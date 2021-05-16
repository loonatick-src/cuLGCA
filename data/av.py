import matplotlib.pyplot as plt
import numpy as np

meta = np.genfromtxt('meta.txt', delimiter=',')

sx = meta[0]; sy = meta[1]
x,y = np.meshgrid(np.arange(0, sx/8, 1), np.arange(0, sy/8, 1))

xn = x * 1 + y * (-0.5)
yn = x * 0 + y * (-0.8660254037844386)

px = np.genfromtxt('avx.csv', delimiter=',')
py = np.genfromtxt('avy.csv', delimiter=',')
ocpy = np.genfromtxt('aocpy.csv', delimiter=',')

print("Average occupancy: ", np.sum(ocpy)/(sx*sy))

npx = np.multiply(px, ocpy)
npy = np.multiply(py, ocpy)

fig, axs = plt.subplots(1, 1)

axs.quiver(xn, yn, npx, npy)
axs.axis('equal')
# axs.set_xlim(100,200)
plt.show()

# length = int(sx*sy/64)
# x = np.reshape(xn, (length, 1))
# y = np.reshape(yn, (length, 1))
# px = np.reshape(npx, (length, 1))
# py = np.reshape(npy, (length, 1))

# val = np.concatenate((x,y,px,py),axis=1)
# np.savetxt("gnuplot.csv", val, 
#               delimiter = ",")