'''

Ray tracing algorithm

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import time

class Plane:
    '''

    Parameters
    ----------
    path : string
      Relative path to an image being added to the scene

    r : array
      Position of the image in spherical coordinates

    '''
    def __init__(self,path='./figures/checker_board.jpg',r=[1,0,0],scale=3):
        self.path = path
        self.r = np.array(r)
        self.img = misc.imread(self.path)/255
        ratio = self.img.shape[0]/self.img.shape[1]
        w = scale
        h = scale*ratio
        self.p1 = np.array((self.r[1]-w/2,self.r[2]-h/2))
        self.p2 = np.array((self.r[1]+w/2,self.r[2]+h/2))
        self.p_x = np.linspace(self.p1[0],self.p2[0],self.img.shape[1])
        self.p_y = np.linspace(self.p1[1],self.p2[1],self.img.shape[0])

    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return (idx)

    def get_color(self,x,y):
        x_ind = self.find_nearest(self.p_x,x)
        y_ind = self.find_nearest(self.p_y,y)
        return (self.img[y_ind][x_ind])
