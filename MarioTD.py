
class MarioTD:
    def __init__(self,env, rev_act_map):
        self.env = env
        self.rev_act_map = rev_act_map

    def step(self,action, reverse=True, render = False, return_state = False, return_reward = False):
        if reverse:
            action = self.rev_act_map[action]
        next_state, reward, done, info = self.env.step(action)
        if render:
            self.env.render()
        obs = f"pos_{info['x_pos']}_{info['y_pos']}"
        if done:
            if info['flag_get']:
                #print("Win")
                obs += "__win"
            else:
                #print(f"Game over at {obs}")
                obs += "__game_over"
        if return_state:
            if return_reward:
                return next_state,obs,reward
            else:
                return next_state,obs
        else:
            return obs
    

    def reset(self,return_state=False):
        state = self.env.reset()
        if return_state:
            return state
        else:
            return "Init"
