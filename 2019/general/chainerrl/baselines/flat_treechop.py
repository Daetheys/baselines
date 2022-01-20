from minerl.herobraine.env_specs.simple_embodiment import SimpleEmbodimentEnvSpec
from minerl.herobraine.hero.handler import Handler
import minerl.herobraine.hero.handlers as handlers
from typing import List
import gym

TC_DOC = """
Treechop infinite env
"""

INFTY = 10**9 #INFINITY

class MineRLTreechopv1ABS(SimpleEmbodimentEnvSpec):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'MineRLTreechopABS-v1'

        super().__init__(*args,
                    max_episode_steps=INFTY,
                    reward_threshold=INFTY,
                    **kwargs)
    def create_server_world_generators(self) -> List[Handler]:
        return [
                handlers.FlatWorldGenerator(generatorString='1;7,2;4;decoration')#"1;7,2,6;1"),
    ]
    def create_agent_start(self) -> List[Handler]:
        return [
            # make the agent start with these items
            handlers.SimpleInventoryAgentStart([
                dict(type="diamond_axe", quantity=10), #We give it an axe to chop faster -> maybe give it a diamond axe if the env is infinite ?
            ])
        ]
    def create_rewardables(self) -> List[Handler]:
        return [
            # reward the agent for touching a gold block (but only once)
            handlers.RewardForCollectingItems([
                dict(type="log", amount=1, reward=1.0),
            ])
        ]
    def create_observables(self):
        return super().create_observables() + [
         handlers.ObservationFromCurrentLocation(),
         handlers.ObservationFromLifeStats()
        ]
    def create_agent_handlers(self) -> List[Handler]:
        return []
    def create_actionables(self) -> List[Handler]:
        return super().create_actionables() + [
            # also allow it to equip the axe
            handlers.EquipAction(["diamond_axe"])
        ]
    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            # Sets time to morning and stops passing of time
            handlers.TimeInitialCondition(False, 23000)
        ]
    def create_server_quit_producers(self):
        return []

    def create_server_decorators(self) -> List[Handler]:
        return []

    # the episode can terminate when this is True
    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold

    def is_from_folder(self, folder: str) -> bool:
        return folder == 'treechopv1'

    def get_docstring(self):
        return TC_DOC
      
from minerl.herobraine.hero.spaces import Dict,Discrete,Box,Tuple
import matplotlib.pyplot as plt
class MineRLTreechopv1(gym.Env):
    def __init__(self,*args,**kwargs):
        import matplotlib.pyplot as plt
        #Unregister and register again (for RLLIB)
        if not('MineRLTreechopABS-v1' in gym.envs.registration.registry.env_specs):
            env_abs = MineRLTreechopv1ABS()
            env_abs.register()
        #Env
        self.env = gym.make('MineRLTreechopABS-v1')
        #Define few methods for MineRL
        self.env.env.create_agent_handlers = lambda a : []
        self.env.env.create_server_quit_producers = lambda a : []
        self.env.max_episode_steps = INFTY
        #Action space after action shaping
        #self.action_space = Box(-1,1,shape=(6,))
        self.action_space = self.env.action_space#gym.spaces.Dict({
            #"forward":gym.spaces.Discrete(2),
            #"jump":gym.spaces.Discrete(2),
            #"camera_left":gym.spaces.Discrete(2),
            #"camera_right":gym.spaces.Discrete(2),
            #"camera_up":gym.spaces.Discrete(2),
            #"camera_down":gym.spaces.Discrete(2)})
        #Tuple([Discrete(2),Discrete(2),Discrete(2),Discrete(2),Discrete(2),Discrete(2)])
        #Observation space is unchanged
        self.observation_space = self.env.observation_space
        #Default action dict
        self.default_action = {"attack" : 1,
                               "left" : 0,
                               "right" : 0,
                               "sneak" : 0,
                               "back" : 0,
                               "sprint":0,
                               "forward":0,
                               "camera_left":0,
                               "camera_right":0,
                               "camera_up":0,
                               "camera_down":0,
                               "jump":0}
        #Time 
        self.time = 0
        #To handle exploration policy
        self.gamma = 0.99 #for the discounted sum of rewards
        self.discounted_reward = 0 #the actual sum dynamically updated
        self.return_threshold = -0.03 #Threshold to switch to exporation policy

        self.camera_angle_hor = 0
        self.camera_angle_ver = 0

        self.camera_angle_ver_threshold = 45

        self.nb_img_input = 3

        #For rendering
        self.r = 0

    def reset(self):
        #Reset the env. For RLLIB the return is computed when the env resets. So we will simulate a reset that won't do anything just for RLLIB to think it has reset
        #The ENV only really resets if time=0 (env has just been created) or if the agent died (I don't know how but it happens sometimes)
        if self.time%(4000*200) == 0 or self.env.done:
            print('FULL RESET')
            o = self.env.reset()
            self.obs_seq = [np.mean(o['pov'].copy(),axis=2) for i in range(self.nb_img_input)]
            return self.compute_obs(o['pov'].copy())
        print('FAST RESET')
        return self.compute_obs(self.obs['pov'].copy())

    def compute_obs(self,new_obs):
        del self.obs_seq[0]
        self.obs_seq.append(np.mean(new_obs,axis=2))
        return np.transpose(np.array(self.obs_seq),(1,2,0))

    def step(self,a_box,exploration_policy=True,always_forward=True):
        #Unpack a_box
        a = a_box.copy()
        #a = {}
        #a["forward"] = int(a_box[0]>=-0.6)
        #a["camera_left"] = int(a_box[1]>=0)
        #a["camera_right"] = int(a_box[2]>=0)
        #a["camera_up"] = int(a_box[3]>=0)
        #a["camera_down"] = int(a_box[4]>=0)
        #a["jump"] = int(a_box[5]>=0.9)
        
        #a["jump"] = 0

        #Generate actions for self.env
        fa = self.default_action.copy()
        #Switch to exploration policy if it's allowed and if the return is too low
        #if self.discounted_reward < self.return_threshold and exploration_policy:
        #    a = self.sample_prandom_action()')
        # --- ACTION SHAPING
        #Reversed Actions
        fa['forward'] = 1-a['forward']
        #fa['jump'] = 1-a['jump']
        fa['jump'] = a['jump']
        #Camera Actions
        angle = 5
        #Cut camera angles too large
        #if self.camera_angle_ver >= self.camera_angle_ver_threshold:
        #    a['camera_down'] = 1
        #if self.camera_angle_ver <= -self.camera_angle_ver_threshold:
        #    a['camera_up'] = 1
        #Compute real camera action
        #fa['camera'] = np.array([(a['camera_up']-a['camera_down'])*angle,(a['camera_right']-a['camera_left'])*angle])
        #Update camera angle
        #self.camera_angle_ver = (self.camera_angle_ver+fa['camera'][0])
        #self.camera_angle_hor = (self.camera_angle_hor+fa['camera'][1])%360
        #Other Actions
        #fa['jump'] = a['jump']
        #Equip axe
        fa['equip'] = 'diamond_axe'
        #Step the environment
        o,r,d,i = self.env.step(a)
        #Updates the return
        #self.discounted_reward = self.gamma*self.discounted_reward + r
        #Penalise jumping
        #r -= fa['jump']*5*10**-2
        #For rendering
        self.r = r
        #Saves last obs
        self.obs = o
        #Increase time
        self.time += 1
        #Artificially asks to reset the environement for RLLIB to compute return and learn
        if self.time % 4000 == 0:
            d = True
        return self.compute_obs(o['pov'].copy()),r,d,i
    def render(self,*args,**kwargs):
        #Renders the env
        out = self.env.render(*args,**kwargs)
        #import matplotlib.pyplot as plt
        #plt.text(20, 20, str(self.r), fontsize=12)
        return out

    def set_eval(self):
        self.env.set_eval()

#Registers the env
try:
    env_abs = MineRLTreechopv1ABS()
    env_abs.register()
except gym.error.Error:
    pass
try:
    del gym.envs.registration.registry.env_specs['MineRLTreechop-v1']
except KeyError:
    pass
from gym.envs.registration import register

register(
    id=f"MineRLTreechop-v1",
    entry_point="flat_treechop:MineRLTreechopv1",
    kwargs={},
)
