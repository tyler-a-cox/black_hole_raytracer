import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as scm
import time
from ray_tracing import Plane

def pol2Cart(r):
    """
    r - <array> [r,theta,phi]
    """
    x = r[0] * np.sin(r[1]) * np.cos(r[2])
    y = r[0] * np.sin(r[1]) * np.sin(r[2])
    z = r[0] * np.cos(r[1])

    return x,y,z

def cart2Pol(x):
    """
    x - <array> [x,y,z]
    """
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    if r > abs(x[2]):
        theta = np.arccos(x[2]/r)
        if abs(r * np.sin(theta)) >= abs(x[0]):
            phi = np.arccos(x[0]/(r*np.sin(theta)))
        else:
            phi = 0
    else:
        theta = phi = 0

    return r,theta,phi

def verlet(y,f,t,h):
    """Integrates motion equations
    y - <array> state vector
    f - <array> derivative function
    t - <float> time
    h - <float> time step distance
    """
    #Half step velocity
    F = f(t,y)
    y[3] += 0.5 * h * F[3]
    y[4] += 0.5 * h * F[4]
    y[5] += 0.5 * h * F[5]
    #Full step position
    y[0] += h*y[3]
    y[1] += h*y[4]
    y[2] += h*y[5]
    #Full step velocity
    F = f(t + h,y)
    y[3] += 0.5 * h * F[3]
    y[4] += 0.5 * h * F[4]
    y[5] += 0.5 * h * F[5]

    return y

def render(res = [160,90],angle = 30,R = 5,D = 100,kR = 150,M = 0,Z = 110,shape = "plane",thiccness = 2,l = 5,h = .1):
    """Renders the image in the scene set up
    res       - <array> number of pixels in the [x,y] directions
    angle     - <float> angle covered by the x axis (degrees)
    R         - <float> radius of lens
    D         - <float> z position of the camera
    kR        - <float> maximum radius
    m         - <float> mass of black hole
    Z         - <float> z position of collision plane
    shape     - <float> shape of the collision plane
    thiccness - <float> thickness of collision plane
    l         - <float> standard length for the shape
    """
    dt = time.time()
    def f(t,y):
        """The derivative function for the particle state
        t - <float> time
        y - <array> state vector [r,theta,phi,r_dot,theta_dot,phi_dot]
        """
        a_r = 2*M/(y[0] - 2*M) * y[3]**2/y[0] + (y[0] - 3*M)*y[4]**2

        if y[0] > 1e-14:
            a_theta = -2*(y[0] - 3*M)/(y[0] - 2*M) * y[3]*y[4]/y[0]
        else:
            a_theta = 0

        return [y[3],y[4],y[5],a_r,a_theta,0]

    def inShape(r,Z,thiccness,l,shape):
        """Finds whether or not the particle is within the collision plane
        r - <array> the [r,theta,phi] position of the particle
        """
        x,y,z = pol2Cart(r)

        inside = False

        if abs(z - Z) < thiccness:
            if shape == "square":
                if -l < x < l and -l < y < l:
                    inside = True
            if shape == "circle":
                if x**2 + y**2 <= l**2:
                    inside = True
            if shape == "F":
                if -3 < x < -1 and -5 < y < 5:
                    inside = True
                elif -3 < x < 3 and 3 < y < 5:
                    inside = True
            if shape == "plane":
                inside = True
            if shape == "image":
                obj = Plane(path = './figures/sagrada-familia.jpg',scale = l)
                if obj.p1[1] < x < obj.p2[1] and obj.p1[0] < y < obj.p2[0]:
                    inside = object.get_color(x,y)
                    print(inside)

                return inside



    #Setup camera
    #--------------------------------------------------------------------------
    CAM = np.zeros((res[1],res[0]))
    alpha = angle * np.pi/180
    beta = alpha * res[1]/res[0]
    #--------------------------------------------------------------------------
    per = -1/(res[0]*res[1])
    m = -1
    for a in np.arange(-alpha/2,alpha/2,alpha/res[0]):
        m += 1
        n = -1
        for b in np.arange(-beta/2,beta/2,beta/res[1]):
            per += 1/(res[0]*res[1])
            n += 1
            #Setting up particles
            #==================================================================
            #Setting up particle positions
            #------------------------------------------------------------------
            x = [R*np.sin(a),R*np.cos(a)*np.sin(b),D - R*np.cos(a)*np.cos(b)]
            r0,theta0,phi0 = cart2Pol(x)
            #------------------------------------------------------------------

            #Setting up particle velocities
            #------------------------------------------------------------------
            r0_dot = -np.sqrt(1 - (np.sin(a)**2 + np.cos(a)**2*np.sin(b)**2))
            theta0_dot = np.sqrt(np.sin(a)**2 + np.cos(a)**2*np.sin(b)**2)/r0
            #------------------------------------------------------------------

            y = [r0,theta0,phi0,r0_dot,theta0_dot,0]
            #==================================================================

            #Running traces
            #------------------------------------------------------------------
            t = 0
            positions = [t,y[0],y[1],y[2]]
            while 1:
                y = verlet(y,f,t,h)
                t += h
                positions.append([t,y[0],y[1],y[2]])

                if inShape(y[:3],Z,thiccness,l,shape) != False:
                    CAM[n,m] += 1#inShape(y[:3],Z,thiccness,l,shape)
                    break
                if y[0] > kR:
                    break
                elif y[0] < 3*M:
                    print(y[0],"vs.",3*M)
                    break


            print("{0}%".format(int(per*1000)/10))
            #CAM[n,m] += 1
    #==========================================================================

    #Output
    #--------------------------------------------------------------------------
    plt.imshow(CAM)
    plt.show()
    dt -= time.time()
    print("Run time: {0}s".format(-dt))
    return CAM

def saveIm(CAM,imName = None):
    if imName == None:
        with open("log.txt",'r+') as rec:
            n = int(rec.read()[-1]) + 1
            rec.write(str(n))
            imName = "image{0}.jpg".format(n)

        scm.imsave(imName,CAM)
