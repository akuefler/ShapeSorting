import gym
from game_settings import SHAPESORT_ARGS1, SHAPESORT_ARGS2
from game import ShapeSorter

class ShapeSorterWrapper(ShapeSorter):
    
    def __init__(self):
        super(ShapeSorterWrapper, self).__init__(
            act_mode=ShapeSorterWrapper.act_mode,
            grab_mode=ShapeSorterWrapper.grab_mode,
            shapes=ShapeSorterWrapper.shapes,
            sizes=ShapeSorterWrapper.sizes,
            n_blocks=ShapeSorterWrapper.n_blocks,
            random_cursor=ShapeSorterWrapper.random_cursor,
            random_holes=ShapeSorterWrapper.random_holes,
            step_size=ShapeSorterWrapper.step_size,
            rot_size=ShapeSorterWrapper.rot_size
        )
    
    @classmethod
    def set_initials(cls, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(ShapeSorterWrapper,k,v)

ShapeSorterWrapper.set_initials(**SHAPESORT_ARGS1)

def register_env():    
    gym.envs.register(
        id= "ShapeSorter-v0",
        #entry_point='rltools.envs.julia_sim:FollowingWrapper',
        entry_point='register_shapesorter:ShapeSorterWrapper',
        timestep_limit=15000,
        reward_threshold=15000,
    )
    
    env = gym.make('ShapeSorter-v0')
    return env
