import numpy as np
import matplotlib.pyplot as plt

def pol2Cart(r):
    """Converts spherical coordinates to cartesian
    r - <array> [r,theta,phi] coordinates
    """
    x = r[0] * np.sin(r[1]) * np.cos(r[2])
    y = r[0] * np.sin(r[1]) * np.sin(r[2])
    z = r[0] * np.cos(r[1])

    return x,y,z

def cart2Pol(x):
    """Converts cartesian coordinates to spherical
    x - <array> [x,y,z] coordinates
    """
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    if r > 0:
        theta = np.arccos(x[2]/r)
        if np.sin(theta) != 0:
            phi = np.arccos(x[0]/np.sqrt(x[0]**2 + x[1]**2))
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

def render(res = [16,9],angle = 30,h = .01,R = 5,D = 10,kR = 150,m = 0,Z = -10,thiccness = 2,l = 30,shape = "circle"):
    """Renders the image for the set up scene
    res       - <array> [x,y] resolution
    angle     - <float> angle coverd by the x axis (degrees)
    h         - <float> time step length
    R         - <float> lens radius
    D         - <float> distance of camera from center
    kR        - <float> maximum light distance
    m         - <float> mass of black hole
    Z         - <float> position of collision plane
    thiccness - <float> thickness of collision plane
    l         - <float> standard image length
    """
    #Setting up functions
    #--------------------------------------------------------------------------
    def f(t,y):
        """Derivative function for motion
        t - <float> time
        y - <array> [r,theta,phi,r_dot,theta_dot,phi_dot] state vector
                    [0,  1  , 2 ,  3  ,    4    ,   5   ]
        """
        a_r = y[0]*y[4]**2
        a_theta = 2*y[3]*y[4]/y[0]
        return [y[3],y[4],0,a_r,a_theta,0]

    def inShape(r,shape,l,thiccness):
        """Determines whether a set of coordinates exist within a shape
        r         - <array> [r,theta,phi] coordinates
        shape     - <string> which shape ("circle"/"square")
        l         - <float> standard length of object
        thiccness - <float> thickness of collision plane
        """
        x,y,z = pol2Cart(r)
        inside = False
        if abs(Z - z) <= thiccness:
            if shape == "circle":
                if np.sqrt(x**2 + y**2) <= l:
                    inside = True

            if shape == "square":
                if -l<x<l and -l<y<l:
                    inside = True

        return inside
    #--------------------------------------------------------------------------

    #Setting up the camera
    #--------------------------------------------------------------------------
    alpha = angle * np.pi/180
    beta = alpha * res[1]/res[0]
    CAM = np.zeros((res[1],res[0]))
    #--------------------------------------------------------------------------

    #Rendering up the scene:
    #--------------------------------------------------------------------------
    m = -1
    per = -1/(res[0]*res[1])
    for a in np.arange(-alpha/2,alpha/2,alpha/res[0]):
        m += 1
        n = -1
        for b in np.arange(-beta/2,beta/2,beta/res[1]):
            per += 1/(res[0]*res[1])
            n += 1
            #Setting particle position:
            x,y,z = -R*np.sin(a),R*np.cos(a)*np.sin(b),D - R*np.cos(a)*np.cos(b)
            r0,theta0,phi0 = cart2Pol([x,y,z])
            #Setting particle velocity:
            r0_dot = -np.sqrt(1 - (np.sin(a)**2 + np.cos(a)**2 * np.sin(b)**2))
            theta_dot = np.sqrt(np.sin(a)**2 + np.cos(a)**2 * np.sin(b)**2)/r0
            #State vector:
            y = [r0,theta0,phi0,r0_dot,theta_dot,0]
            #print(y,f(0,y))

            #Running the traces:
            t = 0
            positions = [[t,r0,theta0,phi0]]
            while 1:
                y[:] = verlet(y,f,t,h)
                t += h
                #print(t,y[0])
                positions.append([t,y[0],y[1],y[2]])
                #Checking kill conditions:
                if inShape(positions[-1][1:],shape,l,thiccness):
                    CAM[n,m] += 1
                    #print("hit!!!!!!!")
                    break
                elif positions[-1][1] > kR:
                    #print("Out of bounds")
                    break
                elif positions[-1][0] <= 3*m:
                    #print("3m")
                    break
            #print(positions[-1][1:])
            CAM[n,m] += 1
        print(per*100,"%")

    plt.imshow(CAM)
    plt.colorbar()
    plt.show()
