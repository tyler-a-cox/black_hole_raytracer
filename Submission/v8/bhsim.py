import prog as pb
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as scm
import time

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

rR = 5

def render(res = [16*rR,9*rR],angle = 30,R = 5,D = 100,kR = 150,M = 0,Z = -10,shape = "image",thiccness = 1,l = .05,h = .1,image = "checker_board.jpg",showHole = False):
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
        color = [0,0,0]

        if "disk" in shape:
            v = [x,y,z]
            rad = float(shape.split()[1])
            nmVr = [1,4,2]
            nmVr /= np.linalg.norm(nmVr)
            if rad * np.dot(v,nmVr) <= thiccness and abs(np.linalg.norm(v) - rad) <= thiccness:
                inside = True
                color = [1,1,1]
        else:
            if abs(z - Z) < thiccness:
                if shape == "square":
                    if -l < x < l and -l < y < l:
                        inside = True
                        color = [1,1,1]
                if shape == "circle":
                    if x**2 + y**2 <= l**2:
                        inside = True
                        color = [1,1,1]
                if shape == "F":
                    if -3 < x/l < -1 and -5 < y/l < 5:
                        inside = True
                        color = [1,1,1]
                    elif -3 < x/l < 5 and 3 < y/l < 5:
                        inside = True
                        color = [1,1,1]
                    elif -3 < x/l < 3 and -1 < y/l < 1:
                        inside = True
                        color = [1,1,1]
                if shape == "plane":
                    inside = True
                    color = [1,1,1]
                if shape == "image":
                    pic = scm.imread("./figures/{0}".format(image))
                    size = [len(pic[0,:]) * l,len(pic[:,0]) * l]
                    if -size[0] < 2*x < size[0] and -size[1] < 2*y < size[1]:
                        inside = True
                        eX = int(x/l - size[0]/2)
                        eY = int(y/l + size[1]/2)
                        color = pic[eY,eX,:]


        return inside,color



    #Setup camera
    #--------------------------------------------------------------------------
    CAM = np.zeros((res[1],res[0],3))
    alpha = angle * np.pi/180
    beta = alpha * res[1]/res[0]
    #--------------------------------------------------------------------------
    per = -1
    m = -1
    for a in np.arange(-alpha/2,alpha/2,alpha/res[0]):
        m += 1
        n = -1
        for b in np.arange(-beta/2,beta/2,beta/res[1]):
            per += 1
            n += 1
            #Setting up particles
            #==================================================================
            #Setting up particle positions
            #------------------------------------------------------------------
            x = [R*np.sin(a),R*np.cos(a)*np.sin(b),D - R*np.cos(a)*np.cos(b)]
            if x[1] > 0:
                x[0] *= -1

            r0,theta0,phi0 = cart2Pol(x)
            #------------------------------------------------------------------

            #Setting up particle velocities
            #------------------------------------------------------------------
            r0_dot = -np.sqrt(1 - (np.sin(a)**2 + np.cos(a)**2*np.sin(b)**2))

            theta0_dot = np.sqrt(np.sin(a)**2 + np.cos(a)**2*np.sin(b)**2)/r0
            if x[1] > 0:
                theta0_dot *= -1
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

                if inShape(y[:3],Z,thiccness,l,shape)[0]:
                    CAM[n,m,:] = inShape(y[:3],Z,thiccness,l,shape)[1]
                    break
                if y[0] > kR:
                    break
                elif y[0] < 3*M:
                    if showHole:
                        CAM[n,m,:] = [1,0,0]
                    break

            #pb.printProgressBar(int(per*1000)/10,100,prefix = 'Rendering...',suffix = 'Complete',length = 50)
            pb.printProgressBar(per + 1,res[0]*res[1],prefix = 'Rendering...',suffix = 'Complete; ETA:{0}s  '.format(int((time.time()-dt)/(per+1)*(res[0]*res[1]-per-1)+1)),length = 50)
            #print("{0}%".format(int(per*1000)/10))
            #CAM[n,m] += 1
    #==========================================================================

    #Output
    #--------------------------------------------------------------------------
    dt -= time.time()
    print("Run time: {0}s".format(-dt))
    plt.imshow(CAM)
    plt.show()
    return CAM

def saveIm(CAM,imName = None):
    if imName == None:
        with open("log.txt",'r+') as rec:
            n = len(rec.read().split(",")) + 1
            m = str(n)
            m += ","
            rec.write(m)
            imName = "image{0}.jpg".format(n)

        scm.imsave(imName,CAM)
