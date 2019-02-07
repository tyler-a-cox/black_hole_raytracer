import matplotlib.pyplot as plt
import numpy as np
import ode

def render(res,angle,R,D,Z,m,kR,l,shape,thiccness = 1,h = 0.01,bcBounds = False,bcCollision = False,bcBlackHole = False):
    """Renders the image in the defined space
    res       - <list> resolution, number of x/y pixels
    angle     - <float> angle covered by x distance
    R         - <float> radius of the "lens" of the camera
    D         - <float> distance of camera from center
    Z         - <float> Z position of the object
    m         - <float> mass of the black hole
    kR        - <float> maximum radius of the particle
    l         - <float> standard shape length
    shape     - <string> name of shape on plane
    thiccness - <float> collision error margin
    h         - <float> lenght of time step
    """

    def pol2Cart(r):
        x = r[1] * np.sin(r[2]) * np.cos(r[3])
        y = r[1] * np.sin(r[2]) * np.sin(r[3])
        z = r[1] * np.cos(r[2])

        return x,y,z

    def cart2Pol(x):
        r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
        if r != 0:
            theta = np.arccos(x[2]/r)
            if np.sin(theta) != 0:
                phi = np.arccos(x[0]/(r*np.sin(theta)))
            else: phi = 0
        else:
            theta = 0
            phi = 0

        return r,theta,phi


    def inShape(r,l,Z,shape,thiccness):
        """The function checking if the coordinates are in the defined shape
        r         - <list> coordinates of the particle
        l         - <float> the standard length of the shape
        Z         - <float> Z position of the shape plane
        shape     - <string> name of the target shape
        thiccness - <float> the thiccness of the target plane
        """

        #We start by converting the position to cartesian coordinates
        x,y,z = pol2Cart(r)

        inside = False

        if abs(z - Z) <= thiccness:
            if shape == "circle":
                if np.sqrt(x**2 + y**2) <= l:
                    inside = True
            elif shape == "square":
                if -l < 2*x < l and -l < 2*y < l:
                    inside = True
            else:
                print("That is a disallowed shape")

        return inside

    def f(t,q):
        """The derivative of the position-velocity list, q
        t - <float> time
        q - <list> position-velocity list [r,theta,phi,r_dot,M,L]
        """
        a_r = 1/q[0]**3 * (q[4]**2 + q[5]**2/np.sin(q[1])**2)
        M_dot = np.cos(q[1])/np.sin(q[1])**3 * q[5]**2/q[0]**2
        L_dot = 0

        r_dot = q[3]
        theta_dot = q[4]/q[0]**2
        phi_dot = q[5]/(q[0]*np.sin(q[1]))**2

        return [r_dot,theta_dot,phi_dot,a_r,M_dot,L_dot]

    #setting up camera properties
    max_alpha = angle/2
    max_beta = angle/2 * res[1]/res[0]
    alpha = np.arange(-max_alpha,max_alpha,2*max_alpha/res[0])
    beta = np.arange(-max_beta,max_beta,2*max_beta/res[1])
    CAM = np.zeros((res[1],res[0]))

    m = -1
    per = -1/(res[0]*res[1])
    for a in alpha:
        m += 1
        n = -1
        for b in beta:
            n += 1
            per += 1/(res[0]*res[1])

            #setting up initial particle position/velocity
            #------------------------------------------------------------------
            x = [0,0,0]
            x[0] = R/np.sqrt(2)*np.sin(a)
            x[1] = R/np.sqrt(2)*np.sin(b)
            x[2] = D - R*np.sqrt(1-(np.sin(a)**2+np.sin(b))/2)
            r0,theta0,phi0 = cart2Pol(x)

            r0_dot = -np.sqrt(1-(np.sin(a)**2 + np.sin(b)**2)/2)
            M0 = r0 * np.sqrt((np.sin(a)**2 + np.sin(b)**2)/2)
            L0 = 0

            q = [r0,theta0,phi0,r0_dot,M0,L0]
            #------------------------------------------------------------------

            t = 0
            while 1:
                positions = []
                positions.append([t,q[0],q[1],q[2]])
                q = ode.velocity_verlet(q,f,t,h)
                t += h
                print(pol2Cart(q[0:4]))

                if inShape(positions[-1],l,Z,shape,thiccness):
                    CAM[n,m] += 1
                    if bcCollision == True:
                        print("Cause of death: Collision.",pol2Cart(positions[-1]))
                    break
                elif positions[-1][1] > kR:
                    if bcBounds == True:
                        print("Cause of death: out of bounds.",pol2Cart(positions[-1]))
                    break
                elif positions[-1][1] <= 3*m:
                    if bcBlackHole == True:
                        print("Cause of death: black hole.",pol2Cart(positions[-1]))
                    break


            #print("The final position is:\nPolar:",positions[-1],"\nCartesian:",pol2Cart(positions[-1]))
        print(per*100,"%")

    plt.imshow(CAM)
    plt.show()
    return CAM
