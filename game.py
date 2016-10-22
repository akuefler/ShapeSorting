import pygame
from pygame import Surface
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy
import matplotlib as mpl

import sys
import gym
from gym.spaces import Discrete, Box

from action_maps import DISCRETE_ACT_MAP4 as DISCRETE_ACT_MAP
#from action_maps import DISCRETE_ACT_MAP3 as DISCRETE_ACT_MAP

from math import pi

import time
import random

WHITE=(255,255,255)
RED = (255,0,0)
OUTLINE=(255,255,0)
GREEN=(0,255,0)
BLUE= (0,0,255)
BLACK= (0,0,0)

TOL = 10
PAD = 500
#DISCRETE_STEP = 20
DISCRETE_STEP = 20

DISCRETE_ROT = 30 #should be 30?

E = 20
#T = 5000
T = 5000

class Block(object):
    def __init__(self, color, center, size, typ):
        self.color=color
        self.center=center
        self.size=size
        self.typ=typ
        self.surface=Surface((size,size))
        self.surface.fill((0,0,255))
                
    def render(self):
        raise NotImplementedError

class Disk(Block): # Something we can create and manipulate
    def __init__(self, color, center, size, typ, angle): # initialze the properties of the object
        Block.__init__(self, color, center, size, typ)
    
    def render(self,screen,angle):
        pygame.draw.circle(screen,self.color,self.center,self.size)
        if self.typ == 'block':
            pygame.draw.circle(screen,OUTLINE,self.center,self.size,1)
            
    def rotate(self,angle):
        pass
        
class PolyBlock(Block): # Something we can create and manipulate
    def __init__(self,color,center,size, typ, angle): # initialze the properties of the object
        Block.__init__(self, color, center, size, typ)
        self.surface=Surface((size+PAD,size+PAD))
        self.surface.fill(WHITE)
        
        self.angle= angle        
    
    def render(self,screen, angle= 5.0):
        self.surface.fill((255,255,255))
        half = self.size/2
        v = self.vertices+self.center
        
        #pygame.draw.rect(screen,self.color,rect)
        pygame.draw.polygon(screen,self.color,v)
        if self.typ == 'block':
            pygame.draw.polygon(screen,OUTLINE,v,1)
            
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
    def __init__(self, color, center, size, typ, angle = 0.0):
        PolyBlock.__init__(self, color, center, size, typ, angle)
        half= self.size/2
        self.vertices=self.V= np.array([[-half,half],[half,half],[half,-half],[-half,-half]])
        
        self.rotate(self.angle)
        
        
class Tri(PolyBlock):
    def __init__(self, color, center, size, typ, angle = 0.0):
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
        

def fit(hole, block):
    pos_fit= np.linalg.norm(np.array(hole.center)-
               np.array(block.center)) < TOL
    
    if not isinstance(block,Disk):
        v = block.vertices.shape[0]
        
        hole_ang = hole.angle - hole.angle
        block_ang= abs(block.angle - hole.angle) % 360
        ang_fit= any([abs(block_ang - ang) < TOL for ang in range(0,360,360/v)])
    else:
        ang_fit = True
    
    #ang_fit= abs(hole.ang - block.ang) < TOL
    return (pos_fit and ang_fit)

def process_observation(screen):

    r = pygame.surfarray.pixels_red(screen).astype('float32')
    g = pygame.surfarray.pixels_green(screen).astype('float32')
    b = pygame.surfarray.pixels_blue(screen).astype('float32')
    X = (2 ** 2) * r + 2 * g + b # convert to 1D map
    Z = imresize(X,(84,84))
    Y = (Z.astype('float32') - 255/2) / (255/2)
    
    return Y
          
def create_renderList(specs, H, W):
    """
    specs= {'disk': {'color':x,'size':y, 'bPostions':z, 'hPosition':w},
            'rect': {},
            'tria': {}}
    """
    blockList = []
    holeList = []
    for key, val in specs.iteritems():
        #if key == 'disk':
            #obj = Disk
        #elif key == 'rect':
            #obj = Rect
        #elif key == 'tri':
            #obj = Tri
        #else:
            #pass
        obj = key
        kwargs = {key : val for key, val in val.iteritems()
                  if key not in ['bPositions', 'hPosition', 'bAngles', 'hAngle']}
        
        for i, per in enumerate(val['bPositions']):
            kwargs['center'] = (int(per[0]*H), int(per[1]*W))
            kwargs['typ'] = 'block'
            if 'bAngles' in val.keys():
                kwargs['angle']= val['bAngles'][i]
            blockList.append(obj(**kwargs))
            
        hPercent = val['hPosition']
        kwargs['center'] = (int(hPercent[0]*H), int(hPercent[1]*W))
        kwargs['typ'] = 'hole'
        kwargs['color'] = BLACK
        #if not obj == Disk:
        kwargs['angle'] = val['hAngle']
        holeList.append(obj(**kwargs))
            
    #renderList = holeList + blockList
    return holeList, blockList
            
class ShapeSorter(object):
    def __init__(self, act_mode= 'discrete', grab_mode= 'toggle',
                 shapes = [Disk, Rect, Tri],
                 sizes = [30, 50, 60],
                 random_cursor= False,
                 random_layout= True,
                 n_blocks = 3,
                 observe_fn= None):
        pygame.init()
        self.H = 200; self.W = 200
        
        self.shapes = shapes
        self.sizes = sizes
        self.n_blocks = n_blocks
        
        self.screen=pygame.display.set_mode((self.H, self.W))
        self.screenCenter = (self.H/2,self.W/2)
        self.act_mode = act_mode
        self.grab_mode= grab_mode
        
        self.random_layout = random_layout
        self.random_cursor = random_cursor
        
        if observe_fn is None:
            self.observe_fn = lambda x : x
        else:
            self.observe_fn = observe_fn
        
        self.initialize()
        
    def initialize(self):
        self.state= {}
        if self.act_mode == 'discrete':
            self.action_space = Discrete(len(DISCRETE_ACT_MAP))            
            self.state['x_speed'] = 0
            self.state['y_speed'] = 0
        else:
            raise NotImplementedError
        
        self.observation_space = Box(0, 1, 84 * 84)
            
        block_selections= np.random.multinomial(len(self.shapes), [1./self.n_blocks]*self.n_blocks)
        bPers = [None]*len(self.shapes)
        hPers = [None]*len(self.shapes)
        bAngs = [None]*len(self.shapes)
        hAngs = [None]*len(self.shapes)
        
        assert len(block_selections) == len(self.shapes)
        
        canonical_positions = [[0.03 * self.H/DISCRETE_STEP,0.03 * self.H/DISCRETE_STEP],
                               [0.07 * self.H/DISCRETE_STEP,0.03 * self.H/DISCRETE_STEP],
                               [0.03 * self.H/DISCRETE_STEP,0.07 * self.H/DISCRETE_STEP],
                               [0.07 * self.H/DISCRETE_STEP,0.07 * self.H/DISCRETE_STEP]
                               ]
        random.shuffle(canonical_positions)
        
        for i, n_b in enumerate(block_selections):
            bPers[i] = np.around(np.random.uniform(0.05,0.95,(n_b,2)),1)
            bAngs[i] = np.random.randint(1,360/DISCRETE_ROT,(n_b,)) * DISCRETE_ROT % 360
            hPers[i] = canonical_positions[i]
            hAngs[i] = np.random.randint(1,360/DISCRETE_ROT) * DISCRETE_ROT % 360
                        
        D = {shape: {'color':RED,
                     'size':self.sizes[i],
                     'bPositions':bPers[i],
                     'hPosition':hPers[i],
                     'bAngles':bAngs[i],
                     'hAngle':hAngs[i]
                     }
            for i, shape in enumerate(self.shapes)
            }
            
        hList, bList = create_renderList(D, self.H, self.W)
        #hList, bList= create_renderList({'disk':{'color':RED, 'size':30,
                                               #'bPositions':pers[0],
                                               #'hPosition' :(0.3,.3)},
                                       #'rect':{'color':RED, 'size':50,
                                               #'bPositions': pers[1],
                                               #'hPosition' :(.7,.3),
                                               #'bAngles' : np.random.randint(1,12,(len(pers[1])
                                                                                   #,)) * DISCRETE_ROT % 360},
                                       #'tri':{'color':RED, 'size':60,
                                              #'bPositions': pers[2],
                                              #'hPosition' : (.5,.7),
                                              #'bAngles' : np.random.randint(1,12,(len(pers[2])
                                                                                  #,)) * DISCRETE_ROT % 360}
                                       #},
                                       #self.H, self.W)          

        #else:
            #hList, bList= create_renderList({'disk':{'color':RED, 'size':30,
                                                   #'bPositions':pers[0],
                                                   #'hPosition' :(0.3,.5)},
                                           #'rect':{'color':RED, 'size':50,
                                                   #'bPositions': pers[1],
                                                   #'hPosition' :(.7,.5)}
                                           #},
                                            #self.H, self.W)       

        self.state['hList'] = hList
        self.state['bList'] = bList
        self.state['grab'] = False
        self.state['target'] = None
        if self.random_cursor:
            self.state['cursorPos'] = np.array([np.random.randint(self.W*0.1, self.W - 0.1*self.W),
                                                                  np.random.randint(self.H*0.1, self.H - 0.1*self.H)])
        else:
            self.state['cursorPos'] = self.screenCenter
        self.state['history']= []        
        #self.state['prevObs']= observation
        
    def step(self, action):
        info = {}
        reward = 0.0
        done = False
        prevCursorPos = self.state['cursorPos']
              
        penalize = False
        self.screen.fill(WHITE)
        
        if type(action) != list:
            if self.act_mode == 'discrete':
                agent_events = DISCRETE_ACT_MAP[action]
            elif self.act_mode == 'continuous':
                raise NotImplementedError
        else:
            agent_events = action
        
        if self.grab_mode != 'toggle':
            if 'grab' in agent_events:
                self.state['grab'] = True
            else:
                self.state['grab'] = False
        else:
            if 'grab' in agent_events:
                self.state['grab'] = not self.state['grab']
            
        if 'left' in agent_events:
            self.state['x_speed'] = x_speed = -DISCRETE_STEP
        elif 'right' in agent_events:
            self.state['x_speed'] = x_speed = DISCRETE_STEP
        else:
            self.state['x_speed'] = x_speed = 0
            
            
        if 'up' in agent_events:
            self.state['y_speed'] = y_speed = -DISCRETE_STEP
        elif 'down' in agent_events:
            self.state['y_speed'] = y_speed = DISCRETE_STEP
        else:
            self.state['y_speed'] = y_speed = 0
            
        (x_pos, y_pos) = self.state['cursorPos']
        self.state['cursorPos'] = cursorPos = (int(max([min([x_pos + x_speed, self.H - 0.1*self.H]),self.H*0.1])),
                                               int(max([min([y_pos + y_speed, self.W - 0.1*self.W]),self.W*0.1])))
        self.state['cursorDis'] = cursorDis = np.array(cursorPos) - np.array(prevCursorPos)        
            
        if 'rotate_cw' in agent_events and self.state['target']:
            self.state['target'].rotate(-DISCRETE_ROT)
            reward += 0.1 / self.n_blocks
            
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, pi/2, 3*pi/2, 15)            
        
        if 'rotate_ccw' in agent_events and self.state['target']:
            self.state['target'].rotate(DISCRETE_ROT)
            reward += 0.1 / self.n_blocks
            
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, 3*pi/2, pi/2, 15)            
        
        #Penalize border hugging:
        if cursorPos[1] == self.W - 0.1*self.W or cursorPos[1] == self.W*0.1 or \
           cursorPos[0] == self.H - 0.1*self.H or cursorPos[0] == self.H*0.1:
            reward -= 0.1 / self.n_blocks
            penalize= True
        
        if self.state['grab']:
            #cursorDis = self.state['cursorDis']
            if self.state['target'] is None:
                for block in self.state['bList']:
                    if isinstance(block,Rect) or isinstance(block,Tri):
                        boundary= block.size/2
                    elif isinstance(block,Disk):
                        boundary= block.size
                    if (prevCursorPos[0]>=(block.center[0]- boundary) and 
                        prevCursorPos[0]<=(block.center[0]+ boundary) and 
                        prevCursorPos[1]>=(block.center[1]- boundary) and 
                        prevCursorPos[1]<=(block.center[1]+ boundary) ): # inside the bounding box
                        
                        self.state['target']=block # "pick up" block
                        self.state['bList'].append(self.state['bList'].pop(self.state['bList'].index(block)))
                        #target.center=cursorPos
                        
            if self.state['target'] is not None:
                self.state['target'].center = tuple(np.array(self.state['target'].center) + cursorDis)
                if not penalize:
                    reward += 0.1 / self.n_blocks
                    
        else:
            if self.state['target'] is not None:
                for hole in self.state['hList']:
                    if type(hole) == type(self.state['target']):
                        if fit(hole, self.state['target']):
                            self.state['bList'].remove(self.state['target'])
                            reward += 1000.0 / self.n_blocks
                        
                    #if (type(hole) == type(self.state['target']) and
                        #np.linalg.norm( np.array(hole.center) 
                                        #- np.array(self.state['target'].center))
                        #< TOL):
                        #self.state['bList'].remove(self.state['target'])
                        #reward += 1000.0 / self.numBlocks
                        
            self.state['target'] = None
            
        for item in self.state['hList'] + self.state['bList']:
            #item.rotate(5.0)
            item.render(self.screen, angle=5.0) # Draw all items
                
        #Render Cursor
        if self.state['grab']:
            col= BLUE
        else:
            col= GREEN 
        pygame.draw.circle(self.screen, col, self.state['cursorPos'], 10)
        #if 'rotate_cw' in agent_events:
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, 2*pi/3, 4*pi/3, 15)
        #if 'rotate_ccw' in agent_events:
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, pi/3, 5*pi/3, 15)             
            
        
        if self.state['bList'] == []:
            done= True
            reward+= 5000.0 / self.n_blocks
        
        #X = np.swapaxes(pygame.surfarray.array3d(self.screen),0,1)
        observation = self.observe_fn(self.screen)
        
        return observation, reward, done, info
    
    def reset(self):
        self.initialize()
        observation, _, _, _ = self.step([])        
        return observation
    
    def render(self):
        pygame.draw.rect(self.screen, BLACK, (self.H*0.1, self.W*0.1,
                                              self.H - 2*self.H*0.1, self.W - 2*self.W*0.1), 1)        
        time.sleep(0.1)
        pygame.display.flip()
            
def main(smooth= False, mode= 'discrete'):
    ss= ShapeSorter(act_mode= mode, observe_fn= process_observation)
    acts_taken = 0
    running = True
    actions= []
    while running:
        ss.reset()
        done= False
        for t in range(T):
            if smooth is False:
                actions= []
               
            flag = False           
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running=False
                    break
                
                if mode == 'discrete':
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            actions.append('grab')
                        
                        #Adjust speed of cursor.
                        if event.key == pygame.K_LEFT:
                            actions.append('left')
                        elif event.key == pygame.K_RIGHT:
                            actions.append('right')
                        elif event.key == pygame.K_UP:
                            actions.append('up')
                        elif event.key == pygame.K_DOWN:
                            actions.append('down')
                            
                        acts_taken += 1
                        #print acts_taken
                        flag= True
                        #print "euc norm: %f, kl norm: %f"%(euc_norm, kl_norm)
                        
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a:
                            actions.append('rotate_ccw')
                        elif event.key == pygame.K_d:
                            actions.append('rotate_cw')
                            
                    if event.type == pygame.KEYUP and smooth:
                        if event.key == pygame.K_SPACE:
                            actions.remove('grab')
                        
                        #Adjust speed of cursor.
                        if event.key == pygame.K_LEFT:
                            actions.remove('left')
                        elif event.key == pygame.K_RIGHT:
                            actions.remove('right')
                        elif event.key == pygame.K_UP:
                            actions.remove('up')
                        elif event.key == pygame.K_DOWN:
                            actions.remove('down')                
              
            if actions == []:
                actions.append('none')
                
            _,reward,done,info = ss.step(actions)
            ss.render()
            
            if done:
                break
    
                    
if __name__ == '__main__':
    X = main(smooth= False, mode= 'discrete') # Execute our main function
