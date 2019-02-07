import numpy as np
import ode
import matplotlib.pyplot as plt

def render(res,angle,R,m,D,shape,l,Z,thickness,K,h = 0.01):
    """Simulates a camera in the setup space
    res - <1x2 array> number of pixels in [x,y]
    angle - <1x2 array> angle of the visible range
    R - <float> the radius of the lens
    m - <float> the mass of the black hole
    D - <float> distance from origin to camera
    shape - <string> the refered to shape
    l - <float> standard length of shape
    Z - <float> z position of the visible surface
    thickness - <float> the thickness of the visible surface
    K - <float> particle kill distance
    h - <float> time steps
    """

    def f(t,q):
        """This function derives the input motion information q"""
        #The acceleration functions
        a_r = 1/q[0]**3*(q[4]**2 + q[5]**2/np.sin(q[1])**2)
        M_dot = np.cos(q[1])/np.sin(q[1])**3 * q[5]**2/q[0]**2
        #Velocity functions
        theta_dot = M/q[0]**2
        phi_dot = q[5]/(q[0]*np.sin(q[1]))**2

        return [q[3],theta_dot,a_r,phi_dot,M_dot,0]

    def inShape(p,shape,l):
        """This function checks if the given input is in the shape"""
        #This translates the positions to cartesian coordinates
        x = p[1]*np.sin(p[2])*np.cos(p[3])
        y = p[1]*np.sin(p[2])*np.sin(p[3])
        z = p[1]*np.cos(p[2])

        if shape == "F":
            if -1.5 < x/l < -0.5 and -2.5 < y/l < 2.5:
                return True
            elif -1.5 < x/l < 1.5 and 1.5 < y/l < 2.5:
                return True
            elif -0.5 < x/l < 0.5 and 0.5 < y/l < 1.5:
                return True
            else:
                return False

        if shape == "square":
            if -0.5 < x/l < 0.5 and -0.5 < y/l < 0.5:
                return True
            else:
                return False

        if shape == "circle":
            if np.sqrt(x**2 + y**2) <= l/2:
                return True
            else:
                return False

    CAM = np.zeros((res[1],res[0]))

    m = -1
    for alpha in np.arange(-angle[0],angle[0],2*angle[0]/res[0]):
        m += 1
        n = -1
        for beta in np.arange(-angle[1],angle[1],2*angle[1]/res[1]):
            n += 1
            #The spherical coordinates for the particle starting at the camera lense
            r0 = np.sqrt(2*R**2*(np.sin(alpha)**2 + np.sin(beta)**2) + D**2 - 2*R*D*np.sqrt(np.sin(alpha)**2 + np.sin(beta)**2))
            theta0 = np.arccos((D - R*np.sqrt(np.sin(alpha)**2 + np.sin(beta)**2))/r0)
            phi0 = np.arccos(R*np.sin(alpha)/(r0*np.sin(theta0)))
            #The trajectory of the particle starting at the camera lense
            r_dot = -np.sqrt(1 - (np.sin(alpha)**2 + np.sin(beta)**2))
            M = r0*np.sqrt(1-r_dot**2)
            L = 0

            q = [r0,theta0,phi0,r_dot,M,L]
            t = 0
            positions = [[t,q[0],q[1],q[2]]]
            while True:
                ode.velocity_verlet(q,f,t,h)
                t += h
                positions.append([t,q[0],q[1],q[2]])

                if q[0] > K:
                    break
                if q[0] < 3*m:
                    break
                if abs(Z - q[0] * np.cos(q[1])) <= thickness:
                    if inShape(q,shape,l):
                        CAM[n,m] += 1
                    break
            CAM[n,m] += 1


    print(len(CAM[0,:]),"x",len(CAM[:,0]),type(CAM),"\n",CAM)
    plt.imshow(CAM)
    plt.show()
