import matplotlib.pyplot as plt
import numpy as np
import ode

def verlet(y,f,t,h):
    r_1 = np.array(y[:3])
    v_0 = np.array(y[3:])

    v_1 = v_0 + (h/2 * f(t,y))[3:]
    r_2 = r_1 + h * v_1

    y1 = r_2
    y1 += v_1

    v_2 = v_1 + (h/2 * f(t + h,y1))[3:]

    y1 = r_2
    y1 += v_2

    return y1

def pol2Cart(r):
    """This function translates spherical coordiantes to cartesian coordiantes
    r - <array> spherical coordinates
    """
    x = r[0] * np.sin(r[1]) * np.cos(r[2])
    y = r[0] * np.sin(r[1]) * np.sin(r[2])
    z = r[0] * np.cos(r[1])

    return x,y,z

def cart2Pol(x):
    """This function translates cartesian coordiantes to spherical coordinates
    x - <array> cartesian coordinates
    """
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    if r > 0:
        theta = np.arccos(x[2]/r)
    else:
        theta = 0

    if not theta in [0,np.pi]:
        phi = np.arccos(x[0]/(r*np.sin(theta)))
    else:
        phi = 0

    return r,theta,phi

def render(res = [16,9],angle = 30.,R = 5.,D = 50,Z = 10,l = 50,shape = "circle",thiccness = 1,m = 0,kR = 100,h = 0.01):
    """This function traces light paths from the camera in the designed environment
    res   - <array> the resolution of the image
    angle - <float> the angle covered by the x axis of the image (degrees)
    R     - <float> the radius of the camera lens
    D     - <float> distance of camera from center
    Z     - <float> z position of collision plane
    m     - <float> mass of the black hole
    kR    - <float> maximum distance for traces to travel
    h     - <float> the time step length
    """
    def inShape(r,Z,l,shape,thiccness):
        """This returns the boolean result of whether or not the coordinates
        are inside of the shape
        r         - <array> spherical coordinates of the particle
        Z         - <float> the z-position of the collision plane
        l         - <float> the standard length for the shape
        shape     - <string> the shape being addressed
        thiccness - <float> the thickness of the collision plane
        """
        inside = False
        x,y,z = pol2Cart(r)

        if abs(z-Z) <= thiccness:
            if shape == "circle":
                if x**2 + y**2 >= l**2:
                    return True
            elif shape == "square":
                if -l < 2*x < l and -l < 2*y < l:
                    return True

        return inside

    def f(t,y):
        """The derivative of the state vector [r,theta,phi,r_dot,M]
        t - <float> time
        y - <array> the state vector
        """
        a_r = y[4]**2/y[0]**3
        theta_dot = y[4]/y[0]**2

        return np.array([y[3],theta_dot,0,a_r,0,0])

    alpha = angle * np.pi/180
    beta = alpha * res[1]/res[0]
    CAM = np.zeros((res[1],res[0]))

    m = -1
    per = -1/(res[0]*res[1])
    for a in np.arange(-alpha/2,alpha/2,alpha/res[0]):
        m += 1
        n = -1
        for b in np.arange(-beta/2,beta/2,beta/res[1]):
            per += 1/(res[0]*res[1])
            n += 1
            #Setup:
            #------------------------------------------------------------------
            #Initializing the positions
            x = [-R*np.sin(a),R*np.cos(a)*np.sin(b),D-R*np.cos(a)*np.cos(b)]
            r0,theta0,phi0 = cart2Pol(x)
            #Initializing the velocities
            r0_dot = -np.sqrt(1 - (np.sin(a)**2 + np.cos(a)**2*np.sin(b)**2))
            M = r0*np.sqrt(np.sin(a)**2 + np.cos(a)**2 * np.sin(b)**2)

            y = [r0,theta0,phi0,r0_dot,M,0]
            #------------------------------------------------------------------
            #Run traces:
            t = 0
            positions = [[t,y[0],[1],y[2]]]
            while 1:
                y = verlet(y,f,t,h)
                t += h
                positions.append([t,y[0],[1],y[2]])

                if y[0] > kR:
                    break
                elif y[0] <= 3*m:
                    break
                elif inShape(y[:3],Z,l,shape,thiccness):
                    CAM[n,m] += 1
                    break
            CAM[n,m] += 1
            print(pol2Cart(positions[-1][1:])[1])
        print(per*100,"%")

    print(CAM)
    plt.imshow(CAM)
    plt.show()
