import ode
import numpy as np
import matplotlib.pyplot as plt

def runTrace(r0,v0,M,L,D,m = 0,Y = -100.,h = 0.01):
    """This runs light traces and finds their end position
    r0 - <array> of [r,theta,phi] starting positions
    v0 - <float> of the initial radial velocity
    L - <float> of the azimuthal momentum
    M - <float> the initial angular momentum
    Y - <float> the Y position of the test plane
    m - <float> the mass of the black hole
    D - <float> the kill distance for the light rays
    """
    y = r0
    y += [v0,M,L]

    def f(t,y):
        """This defines the path of the ray around the black hole
        y = [r,theta,phi,v,M,L]"""

        r_ddot = 1/y[0]**3 * (M**2 + L**2/np.sin(y[1])**2)
        M_dot = np.cos(y[1])/np.sin(y[1])**3 * L**2/y[0]**2
        return [y[3],y[4]/y[0]**2,y[5]/(y[0]**2 * np.sin(y[1])**2),r_ddot,0,M_dot]

    positions = []
    t = 0
    while True:
        place = [t]
        place += y
        positions.append(place)
        y[:] = ode.velocity_verlet(y,f,t,h)
        t += h
        if y[0] > D:
            break
        elif abs(Y - y[0] * np.cos(y[1])) <= 1:
            break
        elif y[0] <= 3*m:
            break

    return positions

def cam(res,angle,m,Y,R = 1,Z = 100.,D = 150.,l = 5,t = 1):
    """This sets up the camera for the scene
    res - <array> number of [x,y] pixels
    R - <float> proportional to the focal length of the camera
    angle - <array> angle of the [x,y] dimensions
    Z - <float> Height of camera from the central coordinate
    D - <float> the kill distance for the light rays
    """

    def inShape(P,Y,t = 1,l = 5):
        """It's an F"""
        inPlane = False

        r = P[1]
        theta = P[2]
        phi = P[3]

        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)

        if abs(z-Y) <= t:
            if -1.5<x/l<1.5 and -2.5<y/l<2.5:
                return True
            elif -1.5<x/l<1.5 and 1.5<y/l<2.5:
                return True
            elif -0.5<x/l<0.5 and 0.5<y/l<1.5:
                return True
            else:
                return False
        else:
            return False

    p = np.arange(-angle[0],angle[0],1/res[0])
    q = np.arange(-angle[1],angle[1],1/res[1])



    CAM = np.zeros((res[1],res[0]))

    n = m = -1
    for alpha in p:
        n += 1
        for beta in q:
            m += 1
            r0 = [np.sqrt(2*R**2*(np.sin(alpha)**2 + np.sin(beta)**2) + Z**2 - 2*R*Z*np.sqrt(np.sin(alpha)**2 + np.sin(beta)**2))]
            r0 += [np.arccos((Z-R*np.sqrt(np.sin(alpha)**2 + np.sin(beta)**2))/r0[0])]
            r0 += [np.arccos(R*np.sin(alpha)/(r0[0]*np.sin(r0[1])))]
            v0 = - np.sqrt(1 - (np.sin(alpha)**2 + np.sin(beta)**2))
            M = r0[0] * np.sqrt(1 - v0**2)
            Ps = runTrace(r0,v0,0,M,D,m,Y = Y)
            if inShape(Ps[-1],Y):
                CAM[alpha,beta] = 1

    plt.imshow(CAM)
    plt.show()
    print(l,"\n",type(CAM),len(CAM[0,:]),"x",len(CAM[:,0]),"\n",CAM)
