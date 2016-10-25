import pygame as pg
from pygame import Surface
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from scipy.stats import entropy
import matplotlib as mpl

import sys
import gym
from gym.spaces import Discrete, Box

from game_settings import DISCRETE_ACT_MAP4 as DISCRETE_ACT_MAP
from game_settings import REWARD_DICT2 as REWARD_DICT

from math import pi

import shapely.geometry
from shape_zoo import *

import time
import random

TOL = 10
E = 20
T = 5000
        
def fit(hole, block):
    """
    determine if block fits in hole.
    """
    pos_fit= np.linalg.norm(np.array(hole.center)-
               np.array(block.center)) < TOL
    U = shapely.geometry.asPolygon(hole.vertices)
    V = shapely.geometry.asPolygon(block.vertices)   

    geom_fit = U.contains(V)
            
    return pos_fit and geom_fit

def process_observation(screen):

    r = pg.surfarray.pixels_red(screen).astype('float32')
    g = pg.surfarray.pixels_green(screen).astype('float32')
    b = pg.surfarray.pixels_blue(screen).astype('float32')
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
    for key, val in specs:

        obj = key
        kwargs = {key : val for key, val in val.iteritems()
                  if key not in ['bPositions', 'hPosition', 'bAngles', 'hAngle']}
        
        for i, per in enumerate(val['bPositions']):
            kwargs['center'] = (int(per[0]*H), int(per[1]*W))
            kwargs['typ'] = 'block'
            if 'bAngles' in val.keys():
                kwargs['angle']= val['bAngles'][i]
            blockList.append(obj(**kwargs))
            
        if val['hDisp']:
            del val['hDisp']
            hPercent = val['hPosition']
            kwargs['center'] = (int(hPercent[0]*H), int(hPercent[1]*W))
            kwargs['typ'] = 'hole'
            kwargs['color'] = BLACK
            #if not obj == Disk:
            kwargs['angle'] = val['hAngle']
            kwargs['size'] = val['size'] + 5
            holeList.append(obj(**kwargs))
            
    #renderList = holeList + blockList
    return holeList, blockList
            
class ShapeSorter(object):
    def __init__(self, act_mode= 'discrete', grab_mode= 'toggle',
                 shapes = [Trapezoid, RightTri, Hexagon, Tri, Rect, Star],
                 #sizes = [50, 60, 40],
                 sizes = [60,60,60,60,60,60],
                 random_cursor= False,
                 random_holes= True,
                 n_blocks = 3,
                 act_map = DISCRETE_ACT_MAP,
                 reward_dict = REWARD_DICT,
                 step_size = 20,
                 rot_size = 30
                 ):
        assert len(sizes) == len(shapes)
        pg.init()
        self.H = 200; self.W = 200
        
        self.act_map = act_map
        self.reward_dict = reward_dict
        
        self.step_size = step_size
        self.rot_size = rot_size
        
        self.shapes = shapes
        self.sizes = sizes
        self.n_blocks = n_blocks
        
        self.screen=pg.display.set_mode((self.H, self.W))
        self.screenCenter = (self.H/2,self.W/2)
        self.act_mode = act_mode
        self.grab_mode= grab_mode
        
        self.random_holes = random_holes
        self.random_cursor = random_cursor
        
        self.initialize()
        
    def initialize(self):
        self.state= {}
        if self.act_mode == 'discrete':
            self.action_space = Discrete(len(self.act_map))            
            self.state['x_speed'] = 0
            self.state['y_speed'] = 0
        else:
            raise NotImplementedError
        
        #self.observation_space = Box(0, 1, 84 * 84)
        self.observation_space = Box(-float('inf'),float('inf'),(84,84))
            
        block_selections= np.random.multinomial(self.n_blocks, [1./len(self.shapes)]*len(self.shapes))
        hDisp = [None]*len(self.shapes)
        bPers = [None]*len(self.shapes)
        hPers = [None]*len(self.shapes)
        bAngs = [None]*len(self.shapes)
        hAngs = [None]*len(self.shapes)
        
        assert len(block_selections) == len(self.shapes)
        
        canonical_positions = [[0.3,0.3],[0.3,0.7],[0.7,0.7],[0.7,0.3]]
        if self.random_holes:
            random.shuffle(canonical_positions)
        
        for i, (shape_ix, n_b) in enumerate(zip(np.argsort(block_selections)[::-1], np.sort(block_selections)[::-1])):
            bPers[shape_ix] = np.around(np.random.uniform(0.05,0.95,(n_b,2)),1)
            #bAngs[shape_ix] = np.random.randint(1,360/self.rot_size,(n_b,)) * self.rot_size % 360
            bAngs[shape_ix] = np.random.randint(0,360/self.rot_size,(n_b,)) * self.rot_size % 360
            
            try:
                hPers[shape_ix] = canonical_positions[i]
                #hAngs[shape_ix] = np.random.randint(1,360/self.rot_size) * self.rot_size % 360
                hAngs[shape_ix] = np.random.randint(0,360/self.rot_size) * self.rot_size % 360                
                hDisp[shape_ix] = True
            except IndexError:
                hPers[shape_ix] = np.array([])
                hAngs[shape_ix] = np.array([])
                hDisp[shape_ix] = False
                        
        D = [(shape, {'color':RED,
                      'hDisp':hDisp[i],
                      'size':self.sizes[i],
                      'bPositions':bPers[i],
                      'hPosition':hPers[i],
                      'bAngles':bAngs[i],
                      'hAngle':hAngs[i]
                    })
            for i, shape in enumerate(self.shapes)
            ]
            
        hList, bList = create_renderList(D, self.H, self.W)      

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
        
    def step(self, action):
        info = {}
        reward = 0.0
        done = False
        prevCursorPos = self.state['cursorPos']
              
        penalize = False
        self.screen.fill(WHITE)
        
        if type(action) != list:
            if self.act_mode == 'discrete':
                agent_events = self.act_map[action]
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
            self.state['x_speed'] = x_speed = -self.step_size
        elif 'right' in agent_events:
            self.state['x_speed'] = x_speed = self.step_size
        else:
            self.state['x_speed'] = x_speed = 0
            
            
        if 'up' in agent_events:
            self.state['y_speed'] = y_speed = -self.step_size
        elif 'down' in agent_events:
            self.state['y_speed'] = y_speed = self.step_size
        else:
            self.state['y_speed'] = y_speed = 0
            
        (x_pos, y_pos) = self.state['cursorPos']
        self.state['cursorPos'] = cursorPos = (int(max([min([x_pos + x_speed, self.H - 0.1*self.H]),self.H*0.1])),
                                               int(max([min([y_pos + y_speed, self.W - 0.1*self.W]),self.W*0.1])))
        self.state['cursorDis'] = cursorDis = np.array(cursorPos) - np.array(prevCursorPos)        
            
        if 'rotate_cw' in agent_events and self.state['target']:
            self.state['target'].rotate(-self.rot_size)
            reward += 0.1 / self.n_blocks
            
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, pi/2, 3*pi/2, 15)            
        
        if 'rotate_ccw' in agent_events and self.state['target']:
            self.state['target'].rotate(self.rot_size)
            reward += 0.1 / self.n_blocks
            
            #shield = (cursorPos[0]-15, cursorPos[1]-15, 30, 30)
            #pygame.draw.arc(self.screen, OUTLINE, shield, 3*pi/2, pi/2, 15)            
        
        #Penalize border hugging:
        if cursorPos[1] == self.W - 0.1*self.W or cursorPos[1] == self.W*0.1 or \
           cursorPos[0] == self.H - 0.1*self.H or cursorPos[0] == self.H*0.1:
            reward += REWARD_DICT['boundary'] / self.n_blocks
            penalize= True
        
        if self.state['grab']:
            #cursorDis = self.state['cursorDis']
            if self.state['target'] is None:
                for block in self.state['bList']:
                    if isinstance(block,PolyBlock):
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
                    #reward += 0.1 / self.n_blocks
                    reward += REWARD_DICT['hold_block'] / self.n_blocks
                    
        else:
            if self.state['target'] is not None:
                dists_and_holes = [(np.linalg.norm(np.array(self.state['target'].center) 
                                                   - np.array(hole.center)),
                                    hole
                                    ) for hole in self.state['hList']]
                hole = min(dists_and_holes)[1]
                if fit(hole, self.state['target']):
                    self.state['bList'].remove(self.state['target'])
                    reward += REWARD_DICT['fit_block'] / self.n_blocks
                        
            self.state['target'] = None
            
        for item in self.state['hList'] + self.state['bList']:
            #item.rotate(5.0)
            item.render(self.screen, angle=5.0) # Draw all items
                
        #Render Cursor
        if self.state['grab']:
            col= BLUE
        else:
            col= GREEN 
        pg.draw.circle(self.screen, col, self.state['cursorPos'], 10)
        
        if self.state['bList'] == []:
            done= True
            reward+= REWARD_DICT['trial_end'] / self.n_blocks
        
        observation = process_observation(self.screen)
        
        return observation, reward, done, info
    
    def reset(self):
        self.initialize()
        observation, _, _, _ = self.step([])        
        return observation
    
    def render(self):
        pg.draw.rect(self.screen, BLACK, (self.H*0.1, self.W*0.1,
                                              self.H - 2*self.H*0.1, self.W - 2*self.W*0.1), 1)        
        time.sleep(0.1)
        pg.display.flip()
            
def main(smooth= False, **kwargs):
    ss= ShapeSorter(**kwargs)
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
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running=False
                    break
                
                if kwargs['act_mode'] == 'discrete':
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_SPACE:
                            actions.append('grab')
                        
                        #Adjust speed of cursor.
                        if event.key == pg.K_LEFT:
                            actions.append('left')
                        elif event.key == pg.K_RIGHT:
                            actions.append('right')
                        elif event.key == pg.K_UP:
                            actions.append('up')
                        elif event.key == pg.K_DOWN:
                            actions.append('down')
                            
                        acts_taken += 1
                        flag= True
                        
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_a:
                            actions.append('rotate_ccw')
                        elif event.key == pg.K_d:
                            actions.append('rotate_cw')
                            
                    if event.type == pg.KEYUP and smooth:
                        if event.key == pg.K_SPACE:
                            actions.remove('grab')
                        
                        #Adjust speed of cursor.
                        if event.key == pg.K_LEFT:
                            actions.remove('left')
                        elif event.key == pg.K_RIGHT:
                            actions.remove('right')
                        elif event.key == pg.K_UP:
                            actions.remove('up')
                        elif event.key == pg.K_DOWN:
                            actions.remove('down')                
              
            if actions == []:
                actions.append('none')
                
            _,reward,done,info = ss.step(actions)
            ss.render()
            
            if done:
                break

if __name__ == '__main__':
    h = Hexagon(RED, (0.,0.), 30, 'block', angle = 0.0)
    from game_settings import SHAPESORT_ARGS0, SHAPESORT_ARGS1, SHAPESORT_ARGS2
    X = main(smooth= False, **SHAPESORT_ARGS2) # Execute our main function
