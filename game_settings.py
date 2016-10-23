from shape_zoo import *

DISCRETE_ACT_MAP1 = \
    [['none'],
     ['up'],
     ['up', 'right'],
     ['right'],
     ['right', 'down'],
     ['down'],
     ['down', 'left'],
     ['left'],
     ['up', 'left'],
     ['grab'],
     ['grab','up'],
     ['grab','up', 'right'],
     ['grab','right'],
     ['grab','right', 'down'],
     ['grab','down'],
     ['grab','down', 'left'],
     ['grab','left'],
     ['grab','up', 'left']     
     ]

DISCRETE_ACT_MAP2 = \
    [['none'],
     ['up'],
     ['up', 'right'],
     ['right'],
     ['right', 'down'],
     ['down'],
     ['down', 'left'],
     ['left'],
     ['up', 'left'],
     ['grab'],    
     ]

DISCRETE_ACT_MAP3 = \
    [['up'],
     ['right'],
     ['down'],
     ['left'],
     ['grab'],    
     ]

DISCRETE_ACT_MAP4 = \
    [['up'],
     ['right'],
     ['down'],
     ['left'],
     ['grab'],    
     ['rotate_cw'],
     ['rotate_ccw']
     ]

REWARD_DICT1 = \
    {'boundary':-0.1,
     'hold_block':0.1,
     'fit_block':1000.0,
     'trial_end':5000.0
    }

REWARD_DICT2 = \
    {'boundary':-0.001,
     'hold_block':0.001,
     'fit_block':10.0,
     'trial_end':50.0
    }

SHAPESORT_ARGS1 = dict(
        _act_mode='discrete',
        _grab_mode='toggle',
        _shapes=[Trapezoid, RightTri, Hexagon, Tri, Rect, Star],
        _sizes=[60,60,60,60,60,60],
        _n_blocks=3,
        _random_cursor=True,
        _random_holes=True
    )