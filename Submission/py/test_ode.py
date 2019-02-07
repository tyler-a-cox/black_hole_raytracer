import numpy as np
from ode import rk4
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.style.use('ggplot')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def sqrnorm(vec):
    return np.einsum('...i,...i',vec,vec)

def RK4f(y,h2):
    f = np.zeros(y.shape)
    f[0:3] = y[3:6]
    f[3:6] = - 1.5 * h2 * y[0:3] / np.power(sqrnorm(y[0:3]),2.5)
    return f

def trace_rays(theta,phi,h=0.01):
    phi = np.deg2rad(phi)
    theta = np.linspace(0,np.deg2rad(theta),15)
    x_p = np.cos(theta)*np.cos(phi)
    y_p = np.sin(theta)*np.cos(phi)
    z_p = np.zeros(theta.shape[0])
    velocities = np.array([x_p,y_p,z_p]).T
    #z_p = np.sin(phi)
    #velocities = np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T
    print (velocities.shape)
    for c,velocity in enumerate(velocities):
        point = np.array([-4.0,0.0,0.0])
        h2 = sqrnorm(np.cross(point,velocity))
        pos = []
        steps = int(500.0/h)
        print (c)
        for i in range(steps):
            y = np.zeros(6)
            y[0:3] = point
            y[3:6] = velocity
            increment = rk4(y,RK4f,h2,h)
            if np.linalg.norm(increment[3:6]) > 5:
                break
            if np.linalg.norm(point+increment[0:3]) < 0.05:
                break
            point += increment[0:3]
            velocity += increment[3:6]
            pos.append(y[0:3])
        pos = np.array(pos)
        plt.plot(pos[:,0],pos[:,1])
    plt.xlim(-4.5,2)
    plt.ylim(-3,3)
    plt.title('Light Rays Traced Around a Black Hole')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('../figures/ray_tracing_bh.png',dpi=300)

trace_rays(39,0,h=0.01)
