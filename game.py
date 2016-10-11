import pygame
from pygame import Surface
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy

import sys
import gym
from gym.spaces import Discrete, Box

from action_maps import DISCRETE_ACT_MAP4 as DISCRETE_ACT_MAP
#from action_maps import DISCRETE_ACT_MAP3 as DISCRETE_ACT_MAP

from habituation import SurprisalManager

from math import pi

import time

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
    def __init__(self, color, center, size, typ): # initialze the properties of the object
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
          
def create_renderList(specs, H, W):
    """
    specs= {'disk': {'color':x,'size':y, 'bPostions':z, 'hPosition':w},
            'rect': {},
            'tria': {}}
    """
    blockList = []
    holeList = []
    for key, val in specs.iteritems():
        if key == 'disk':
            obj = Disk
        elif key == 'rect':
            obj = Rect
        elif key == 'tri':
            obj = Tri
        else:
            pass
        kwargs = {key : val for key, val in val.iteritems()
                  if key not in ['bPositions', 'hPosition', 'bAngles']}
        
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
        if not obj == Disk:
            kwargs['angle'] = 0.0
        holeList.append(obj(**kwargs))
            
    #renderList = holeList + blockList
    return holeList, blockList
            
class ShapeSorter(object):
    def __init__(self, act_mode= 'discrete', grab_mode= 'toggle',
                 include_tris= True,
                 random_cursor= False, random_layout= True):
        pygame.init()
        self.H = 200; self.W = 200
        
        self.screen=pygame.display.set_mode((self.H, self.W))
        self.screenCenter = (self.H/2,self.W/2)
        self.act_mode = act_mode
        self.grab_mode= grab_mode
        self.include_tris=include_tris
        self.sm = SurprisalManager()
        
        self.random_layout = random_layout
        self.random_cursor = random_cursor
        
        self.initialize()
        
    def initialize(self):
        self.state= {}
        if self.act_mode == 'discrete':
            #x_speed= 0
            #y_speed= 0
            self.action_space = Discrete(len(DISCRETE_ACT_MAP))            
            self.state['x_speed'] = 0
            self.state['y_speed'] = 0
        else:
            raise NotImplementedError
        
        #self.observation_space = Box(0, 255, shape=(self.H,self.W,3))
        self.observation_space = Box(0, 1, 84 * 84)
        
        #self.numBlocks= np.random.randint(1,5)
        self.numBlocks= 3
        if self.include_tris:
            block_selections= np.random.multinomial(self.numBlocks, [0.33,0.33,0.33])
            pers= [None, None, None]
        else:
            block_selections= np.random.multinomial(self.numBlocks, [0.5,0.5])
            pers= [None, None]
            
        if self.random_layout:
            for i, b in enumerate(block_selections):
                pers[i]= np.random.uniform(0.05,0.95, (b,2))
                pers[i]= np.around(pers[i],1)

            #diskPer= np.random.uniform(0.05,0.95, (numDisks,2))
            #rectPer= np.random.uniform(0.05,0.95, (numRects,2))
            
            #diskPer= np.around(diskPer,1)
            #rectPer= np.around(rectPer,1)
            
            #numTris = np.random.randint(0,4)
            
            #triPer= np.random.uniform(0.05,0.95, (numTris,2))
            #triPer= np.around(triPer,1)          
            
        else:
            diskPer= [[.4,.7],[.5, .4]]
            rectPer= [(.8,.8)]
            
        if self.include_tris:
            hList, bList= create_renderList({'disk':{'color':RED, 'size':30,
                                                   'bPositions':pers[0],
                                                   'hPosition' :(0.3,.3)},
                                           'rect':{'color':RED, 'size':50,
                                                   'bPositions': pers[1],
                                                   'hPosition' :(.7,.3),
                                                   'bAngles' : np.random.randint(1,12,(len(pers[1])
                                                                                       ,)) * 30 % 360},
                                           'tri':{'color':RED, 'size':60,
                                                  'bPositions': pers[2],
                                                  'hPosition' : (.5,.7),
                                                  'bAngles' : np.random.randint(1,12,(len(pers[2])
                                                                                      ,)) * 30 % 360                                                  }},
                                            self.H, self.W)          

        else:
            hList, bList= create_renderList({'disk':{'color':RED, 'size':30,
                                                   'bPositions':pers[0],
                                                   'hPosition' :(0.3,.5)},
                                           'rect':{'color':RED, 'size':50,
                                                   'bPositions': pers[1],
                                                   'hPosition' :(.7,.5)}
                                           },
                                            self.H, self.W)       

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
            reward += 0.1 / self.numBlocks
            
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, pi/2, 3*pi/2, 15)            
        
        if 'rotate_ccw' in agent_events and self.state['target']:
            self.state['target'].rotate(DISCRETE_ROT)
            reward += 0.1 / self.numBlocks
            
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, 3*pi/2, pi/2, 15)            
        
        #Penalize border hugging:
        if cursorPos[1] == self.W - 0.1*self.W or cursorPos[1] == self.W*0.1 or \
           cursorPos[0] == self.H - 0.1*self.H or cursorPos[0] == self.H*0.1:
            reward -= 0.1 / self.numBlocks
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
                    reward += 0.1 / self.numBlocks
                    
        else:
            if self.state['target'] is not None:
                for hole in self.state['hList']:
                    if type(hole) == type(self.state['target']):
                        if fit(hole, self.state['target']):
                            self.state['bList'].remove(self.state['target'])
                            reward += 1000.0 / self.numBlocks
                        
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
            reward+= 5000.0 / self.numBlocks
        
        X = np.swapaxes(pygame.surfarray.array3d(self.screen),0,1)
        
        r = (pygame.surfarray.pixels_red(self.screen) / 255) * 1
        g = (pygame.surfarray.pixels_green(self.screen) / 255) * 3
        b = (pygame.surfarray.pixels_blue(self.screen) / 255) * 7
        observation = (r + g + b)
        observation = imresize(observation.T, (84,84)).astype('float32')
        
        observation = np.abs(observation - 11.0)
        observation /= 11.0        
        
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
            
def main(smooth= False):
    mode= 'discrete'
    ss= ShapeSorter(act_mode= mode, include_tris= True)
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
            #print reward
            #if flag: #print "euc norm: %f, kl norm: %f"%(info['euc'], info['kl'])
                #for key, val in info.iteritems():
                    #print key,'...',val
            ss.render()
            
            if done:
                break
                
    
#def main(mode= 'continuous'): # Where we start
    #pygame.init()
    #clock = pygame.time.Clock()
    #screen=pygame.display.set_mode((500,500))
    #screenCenter = (250,250)
    #if mode == 'discrete':
        #x_speed= 0
        #y_speed= 0
        
    #running=True
                 
    #hList, bList= create_renderList({'disk':{'color':RED, 'size':50,
                                           #'bPositions':[(300,300)],
                                           #'hPosition' :(200,250)},
                                   #'rect':{'color':RED, 'size':50,
                                           #'bPositions':[(400,100)],
                                           #'hPosition' :(300,250)}})
    #renderList = hList + bList
    #grab= False
    #target=None # target of Drag/Drop
    
    #cursorPos = screenCenter
    #while running:
    ##for _ in range(5000):
        
        #screen.fill((255,255,255)) # clear screen
        #if mode == 'continuous':
            #cursorPos=pygame.mouse.get_pos()
        
        #for event in pygame.event.get():
            #if event.type == pygame.QUIT:
                #running=False
                #break # get out now

            #if mode == 'continuous':
                #if event.type == pygame.MOUSEBUTTONDOWN:
                    #grab = True 
                
                #if event.type == pygame.MOUSEBUTTONUP:
                    #grab = False
                
            #elif mode == 'discrete':
                #if event.type == pygame.KEYDOWN:
                    ##Grab object
                    #if event.key == pygame.K_SPACE:
                        #grab = True
                    
                    ##Adjust speed of cursor.
                    #if event.key == pygame.K_LEFT:
                        #x_speed = -3
                    #elif event.key == pygame.K_RIGHT:
                        #x_speed = 3
                    #elif event.key == pygame.K_UP:
                        #y_speed = -3
                    #elif event.key == pygame.K_DOWN:
                        #y_speed = 3
     
                ## User let up on a key
                #elif event.type == pygame.KEYUP:
                    #if event.key == pygame.K_SPACE:
                        #grab = False
                        
                    ## If it is an arrow key, reset vector back to zero
                    #if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        #x_speed = 0
                    #elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        #y_speed = 0
         
        #if mode == 'discrete':        
            #(x_pos, y_pos) = cursorPos
            #cursorPos = (x_pos + x_speed, y_pos + y_speed)            
                        
        #if grab==True:
            #for item in renderList: # search all items
                #if (item.typ == 'block' and
                        #cursorPos[0]>=(item.pos[0]-item.size) and 
                        #cursorPos[0]<=(item.pos[0]+item.size) and 
                        #cursorPos[1]>=(item.pos[1]-item.size) and 
                        #cursorPos[1]<=(item.pos[1]+item.size) ): # inside the bounding box
                    #target=item # "pick up" item
                        
                #if grab and target is not None: # if we are dragging something
                    #target.pos=cursorPos # move the target with us
                    
        #else:
            #if target is not None:
                #for hole in hList:
                    #if (type(hole) == type(target) and
                        #np.linalg.norm( np.array(hole.pos) - np.array(target.pos))
                        #< TOL):
                        ##print "HOORAY!"
                        #renderList.remove(target)
                        
            #target = None
                
        #for item in renderList:
            #item.render(screen) # Draw all items
        
        ##Render Cursor
        #if grab:
            #col= BLUE
        #else:
            #col= GREEN
        #pygame.draw.circle(screen, col, cursorPos, 10)
        
        #pygame.display.flip()
        
        ##clock.tick(60)
        
        #X = pygame.surfarray.array3d(screen)
        
    #return X# End of function
                    
#if __name__ == '__main__': # Are we RUNNING from this module?
    #X = main() # Execute our main function

#E= 1000
#T= 50

#def gymLoop(env, render):
    #for i_episode in range(E):
        #done = False
        #_ = env.reset()
        #total_cost = 0.0
        #total_reward=0.0
        
        #t = 0
        #while t < T:
            #if render:
                #env.render()
            #action = env.action_space.sample()
            ##print "action: ", DISCRETE_ACT_MAP[action]
            #observation, reward, done, info = env.step(action)
            #print reward
            #total_reward += reward        
            #if done:
                #print("Episode finished after {} timesteps".format(t+1))
                #break
            #t += 1

#env = ShapeSorter(random_cursor= False)
#for ren in [True]:
    ##t1 = time.time()
    #gymLoop(env,ren)
    ##t2 = time.time()
    ##print "Time: ", t2-t1
    
#halt= True