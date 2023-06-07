import numpy as np
import matplotlib.pyplot as plt

nx = 41
ny = 41
nt = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = .01

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
u = np.ones((ny, nx))
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
fig = plt.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, y)

for n in range(nt):
    un = u.copy()
    for j in range(1, ny):
        for i in range(1, nx):
            u[j, i] = un[j, i] - c * dt / dx * (un[j, i] - un[j, i - 1])\
                               - c * dt / dy * (un[j, i] - un[j - 1, i])
    ax.plot_surface(X, Y, u[:], cmap=plt.cm.coolwarm)
    ax.set_zlim(1, 2)
    plt.pause(.01)
    ax.cla()
plt.savefig('figure.jpg')
