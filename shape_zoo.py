import numpy as np
import pygame as pg

PAD = 500

WHITE=(255,255,255)
RED = (255,0,0)
OUTLINE=(255,255,0)
GREEN=(0,255,0)
BLUE= (0,0,255)
BLACK= (0,0,0)

class Block(object):
    def __init__(self, color, center, size, typ):
        self.color=color
        self.center=center
        self.size=size
        self.typ=typ
        self.surface=pg.Surface((size,size))
        self.surface.fill((0,0,255))
                
    def render(self):
        raise NotImplementedError

class Disk(Block): # Something we can create and manipulate
    def __init__(self, color, center, size, typ, angle): # initialze the properties of the object
        Block.__init__(self, color, center, size, typ)
    
    def render(self,screen,angle):
        pg.draw.circle(screen,self.color,self.center,self.size)
        if self.typ == 'block':
            pg.draw.circle(screen,OUTLINE,self.center,self.size,1)
            
    def rotate(self,angle):
        pass
        
class PolyBlock(Block): # Something we can create and manipulate
    def __init__(self,color,center,size, typ, angle): # initialze the properties of the object
        Block.__init__(self, color, center, size, typ)
        self.surface=pg.Surface((size+PAD,size+PAD))
        self.surface.fill(WHITE)
        
        self.angle= angle        
    
    def render(self,screen, angle= 5.0):
        self.surface.fill((255,255,255))
        half = self.size/2
        v = self.vertices+self.center
        
        pg.draw.polygon(screen,self.color,v)
        if self.typ == 'block':
            pg.draw.polygon(screen,OUTLINE,v,1)
            
    def rotate(self, angle):
        self.angle = (self.angle +  angle) % 360
        if self.angle == 0:
            halt= True
        #print self.ang
        
        theta = np.radians(self.angle)
        R = np.array([[np.cos(theta),-np.sin(theta)],
                      [np.sin(theta),np.cos(theta)]])
        self.vertices= np.dot(self.V,R).astype('int64')
               
class Rect(PolyBlock):
    def __init__(self, color, center, size, typ, angle = 0.0, **kwargs):
        PolyBlock.__init__(self, color, center, size, typ, angle)
        third= self.size/3
        self.vertices=self.V= np.array([[-third,third],[third,third],[third,-third],[-third,-third]])
        
        self.rotate(self.angle)
        
        
class Tri(PolyBlock):
    def __init__(self, color, center, size, typ, angle = 0.0, **kwargs):
        """
        size = (length,)
        """
        PolyBlock.__init__(self, color, center, size, typ, angle)
        l = size
        a = np.sqrt((l**2) - ((l**2)/4.0))
        v= np.array([[-l/2.0, 0],[l/2.0, 0],[0, a]])
        
        v -= v.mean(axis= 0)
        self.vertices = self.V = v
        self.rotate(self.angle)
        
class Hexagon(PolyBlock):
    def __init__(self, color, center, size, typ, angle = 0.0, **kwargs):
        PolyBlock.__init__(self, color, center, size, typ, angle)
        third= self.size/3
        
        u = np.ones((1,2)) * third
        thetas = [np.radians(theta) for theta in range(0,360,360/6)]
        R = lambda x : np.array([[np.cos(x),-np.sin(x)],
                                 [np.sin(x),np.cos(x)]])
        self.vertices= self.V = np.row_stack([np.dot(u,R(theta)) for theta in thetas])
        self.rotate(angle)
        
class Star(PolyBlock):
    def __init__(self, color, center, size, typ, angle = 0.0, **kwargs):
        PolyBlock.__init__(self, color, center, size, typ, angle)
        third = int(self.size/3)
        sixth= int(self.size/6)
        
        u = np.ones((1,2))
        thetas = [np.radians(theta) for theta in range(0,360,360/10)]
        magnitudes = np.array([third, sixth] * 5)
        R = lambda x : np.array([[np.cos(x),-np.sin(x)],
                                 [np.sin(x),np.cos(x)]])
        self.vertices= self.V = np.row_stack([np.dot(u,R(theta)) * magnitudes[i] for i, theta in enumerate(thetas)])
        self.rotate(angle)
        
class RightTri(PolyBlock):
    def __init__(self, color, center, size, typ, angle = 0.0, **kwargs):
        PolyBlock.__init__(self, color, center, size, typ, angle)
        
        v = np.array([[0.,0.],[0.,size],[size,0.]])
        v -= v.mean(axis=0)
        self.vertices = self.V = v
        self.rotate(angle)
      
class Trapezoid(PolyBlock):  
    def __init__(self, color, center, size, typ, angle = 0.0, **kwargs):
        PolyBlock.__init__(self, color, center, size, typ, angle)
        half = size/2
        v = np.array([[0.,0.],
                      [size,0.],
                      [int(0.66 * size),int(0.66*size)],
                      [int(0.33 * size),int(0.66*size)]])
        v -= v.mean(axis=0)
        self.vertices = self.V = v
        self.rotate(angle)