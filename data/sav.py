import matplotlib.pyplot as plt
import numpy as np

# print function for getting data
# to matlab, for quiver plot
def print_m(xn):
    for i in xn:
     for j in i:
      print(j, ', ', end='')
     print(';')
    return


meta = np.genfromtxt('meta.txt', delimiter=',')

sx = meta[0]; sy = meta[1]

px = np.genfromtxt('fpx.csv', delimiter=',')
py = np.genfromtxt('fpy.csv', delimiter=',')
ocpy = np.genfromtxt('foccupancy.csv', delimiter=',')

# print("Average occupancy: ", np.sum(ocpy)/(sx*sy))

stride = 8
basex = stride
basey = stride

# print(sy/basey, sx/basex )
avx = np.zeros(((int(sy/basey),int(sx/basex))))
avy = np.zeros(((int(sy/basey),int(sx/basex))))
aoc = np.zeros(((int(sy/basey),int(sx/basex))))

for i in range(int(sy/basex)):
    for j in range(int(sx/basey)):
        avx[i,j] = np.sum(px[i*basex:(i+1)*basex, j*basey:(j+1)*basey])/(basex*basey)
        avy[i,j] = np.sum(py[i*basex:(i+1)*basex, j*basey:(j+1)*basey])/(basex*basey)
        aoc[i,j] = np.sum(ocpy[i*basex:(i+1)*basex, j*basey:(j+1)*basey])/(basex*basey)

npx = np.multiply(avx, aoc)
npy = np.multiply(avy, aoc)

x,y = np.meshgrid(np.arange(0, sx/basex, 1), np.arange(0, sy/basey, 1))
xn = x * 1 + y * (-0.5)
yn = x * 0 + y * (-0.8660254037844386)

# print('xn = [')
# print_m(xn)
# print(']')
# print('yn = [')
# print_m(yn)
# print(']')
# print('px = [')
# print_m(avx)
# print(']')
# print('py = [')
# print_m(avy)
# print(']')
# print('quiver(xn, yn, px, py)')

fig, axs = plt.subplots(1, 1)

axs.quiver(xn, yn, avx, avy)
axs.axis('equal')
# axs.set_xlim(10,50)   # 4
# # axs.set_xlim(5, 20)
# axs.set_xlim(2.5, 12.5)
plt.show()
