from schedulers import PrismInterface
from MarioTD import MarioTD
from aalpy.utils import load_automaton_from_file
import gym_super_mario_bros
import os
import pickle
import resource
import configparser
import datetime
import torch
import numpy as np
from math import sqrt, log
import time
from pathlib import Path
import json
import sys
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import ResizeObservation, SkipFrame
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions
import random
from agent import Mario
import aalpy.paths
from schedulers import extract_coords
from scipy.stats import fisher_exact
aalpy.paths.path_to_prism = "/home/mtappler/Programs/prism-4.7-linux64/bin/prism"
max_rec = 0x20000
action_dim = 2

# May segfault without this line. 0x100 is a guess at the size of each stack frame.
resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)
from enum import Enum
class Verdict(Enum):
    PASS = 1
    FAIL = 2
    INCONC = 3

def save(x, path):
    with open(path, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as handle:
            return pickle.load(handle)
    else:
        return None

def setup_scheduler(model_file_name,target,mdp=None):
    prism_int_name = model_file_name.replace(".dot", f"p_int_{target}.pickle")
    interface = load(prism_int_name)
    if interface is None:
        if mdp is None:
            model = load_automaton_from_file(model_file_name, "mdp")
        else:
            model = mdp
        print("Loaded MDP")
        interface = PrismInterface(target, model)
        print("Scheduler initialized")
        save(interface,prism_int_name)

    scheduler = interface.scheduler
    print("Scheduler initialized")
    return scheduler

def setup_reachability(model_file_name,target,stage,params):
    scheduler = setup_scheduler(model_file_name,target)
    mario = setup_mario(params)
    return mario, scheduler 

def setup_mario(params):
    stage = params["SETUP"]["STAGE"]
    style = params["SETUP"]["STYLE"]
    env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-{style}")
    # due to an episode limit, make in the above line returns TimeLimit environment,
    # so to get the mario environment directly, we need to unwrap
    unwrapped_env = env.env

    # Limit the action-space
    action_space = {
        'SIMPLE_MOVEMENT': JoypadSpace(env, actions.SIMPLE_MOVEMENT),
        'COMPLEX_MOVEMENT': JoypadSpace(env, actions.COMPLEX_MOVEMENT),
        'RIGHT_ONLY': JoypadSpace(env, actions.RIGHT_ONLY),
        'FAST_RIGHT': JoypadSpace(env, [['right','B'], ['right', 'A','B']])
    }
    env = action_space.get("FAST_RIGHT")
    
    # Apply Wrappers to environment
    env = SkipFrame(env, skip_min=3, skip_max=5)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    rev_act_map = {"right":0, "jump":1}
    mario = MarioTD(env,rev_act_map)
    return mario

def setup_test_schedulers(model_file_name, x_positions):
    def closest_label_coord(all_label_coords, x_pos):
        min_diff = 10**10
        min_oc = None
        for (o,c) in all_label_coords:
            x,y = c
            if abs(x-x_pos) < min_diff:
                min_diff = abs(x-x_pos)
                min_oc = (o,c)
        return min_oc
            

    model = load_automaton_from_file(model_file_name, "mdp")
    all_labels = [s.output.replace("__game_over","").replace("__win","") for s in model.states]
    all_labels.remove("Init")
    all_label_coords =  [(o,extract_coords(o)) for o in all_labels]
    schedulers = dict()
    for x in x_positions:
        closest = closest_label_coord(all_label_coords,x)
        target, coord = closest
        scheduler = setup_scheduler(model_file_name,target,mdp=model)
        schedulers[x] = scheduler
    return schedulers

def setup_suts(params):
    suts_list = []
    suts_names = tuple(json.loads(params["TESTING"]["SUTs"]))
    print(suts_names)
    for sut_name in suts_names:
        save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        import time
        time.sleep(2)
        save_dir.mkdir(parents=True)
        
        checkpoint_path = sut_name
        checkpoint = Path(checkpoint_path) 
        sut = Mario(state_dim=(4, 84, 84), action_dim=action_dim, save_dir=save_dir, params=params,
                  checkpoint=checkpoint,load_only_conv=False,disable_cuda=True)
        suts_list.append(sut)
    return suts_names,tuple(suts_list)

def run_single_test(scheduler,x_goal, sut, mario_td,test_len):
    scheduler.reset()
    mario_td.reset()

    while True: 
        action = scheduler.get_input()
        if action is None:
            #print(f"Sampling a random action in {obs}")
            action = random.choice(["right","jump"])
        rl_state,obs = mario_td.step(action,render=False,return_state=True)
        if "game_over" in obs:
            return Verdict.INCONC
        x,y = extract_coords(obs)
        if x >= x_goal:
            break
        reached_state = scheduler.step_to(action, obs)
        if reached_state is None:
            scheduler.step_to_closest(action,obs)

    rl_state = torch.from_numpy(np.array(rl_state)).float()
        
    for i in range(test_len):
        action = sut.act(rl_state,eval_mode=True)
        rl_state,obs = mario_td.step(action,reverse=False,render=False,return_state=True)
        rl_state = torch.from_numpy(np.array(rl_state)).float()
        
        if "game_over" in obs:
            return Verdict.FAIL
        if "win" in obs:
            return Verdict.PASS
    return Verdict.PASS

def diff_test(f1, n1, f2, n2,eps):
    contingency_table = np.array([[f1,n1-f1],[f2,n2-f2]])
    res = fisher_exact(contingency_table, alternative='two-sided')
    if res[1] < eps:
        return True,res
    else:
        return False
    #if abs(f1 / n1 - f2 / n2) > ((sqrt(1 / n1) + sqrt(1 / n2)) * sqrt(0.5 * log(2 / eps))):
    #    return True
    #return False


def repeated_test_from_state(params,x_goal,scheduler,sut_names,suts,mario_td):
    max_tries = params.getint("TESTING","MAX_TRIES")
    eps = params.getfloat("TESTING","ALPHA")
    test_len = params.getint("TESTING","LENGTH")
    
    
    fails_sut_1 = 0
    succ_sut_1 = 0
    succ_sut_2 = 0
    fails_sut_2 = 0
    current_sut_index = 0
    
    for i in range(max_tries):
        if i % 20 == 0:
            print(f"Try: {i}")
        sut = suts[current_sut_index]
        verdict = run_single_test(scheduler,x_goal, sut, mario_td,test_len)
        if verdict == Verdict.INCONC:
            continue
        elif verdict == Verdict.PASS:
            if current_sut_index == 0:
                succ_sut_1 += 1
            else:
                succ_sut_2 += 1
            
            current_sut_index = (current_sut_index + 1) % 2
        else:
            if current_sut_index == 0:
                fails_sut_1 += 1
            else:
                fails_sut_2 += 1
            
            current_sut_index = (current_sut_index + 1) % 2
        # perform hoeffding test here
        n1 = fails_sut_1 + succ_sut_1
        n2 = fails_sut_2 + succ_sut_2
        diff_res = diff_test(fails_sut_1, n1,fails_sut_2,n2,eps)
        if n1 > 0 and n2 > 0 and diff_res:
            print("Stopping early")
            print(f"{fails_sut_1/n1} vs {fails_sut_2/n2}")
            return(i,fails_sut_1,n1,fails_sut_2,n2, diff_res[1])
    print("Similarly safe")
    #return(i,fails_sut_1,n1,fails_sut_2,n2)
    return(max_tries + 1,fails_sut_1,n1,fails_sut_2,n2)
    
    

def differential_testing(params):
    stage = params["SETUP"]["STAGE"]
    model_file_name = params["TESTING"]["MODEL"]
    x_positions = json.loads(params["TESTING"]["X_POSITIONS"])
    print(x_positions)
    for x in x_positions:
        print(x, type(x))
    schedulers = setup_test_schedulers(model_file_name, x_positions)
    mario_td = setup_mario(params)
    sut_names,suts = setup_suts(params)
    result_dict = dict()
    import random
    id = random.randint(0,10000)
    with open(f"test_result_{stage}_{id}.txt", "w") as fp:
        fp.write(str(sut_names))
        fp.write("\n")
        for x in schedulers.keys():
            print(f"Going to test for {x}")
            start = time.time()
            result = repeated_test_from_state(params,x,schedulers[x],sut_names,suts,mario_td)
            end = time.time()
            fp.write(str(result))
            fp.write("\n")
            fp.write(f"Time: {end-start}")
            fp.write("\n")
            fp.flush()
            result_dict[x] = result
    print(result_dict)

def test_main(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    differential_testing(params)

def eval_single(mario_td,sut, n_eval):
    rewards = []
    for i in range(n_eval):
        if i % 10 == 0:
            print(f"Eval: {i}")
        single_reward = 0
        rl_state = mario_td.reset(return_state=True)
        while True:
            rl_state = torch.from_numpy(np.array(rl_state)).float()
            action = sut.act(rl_state,eval_mode=True)
            rl_state,obs,rew = mario_td.step(action,reverse=False,render=False,return_state=True,return_reward=True)
            single_reward += rew
            if "game_over" in obs or "win" in obs:
                break
        rewards.append(single_reward)
    mean_reward = sum(rewards)/len(rewards)
    std_dev = sqrt(sum(map(lambda r : (r - mean_reward)**2,rewards)) / len(rewards))
    return mean_reward, std_dev

def eval_main(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    stage = params["SETUP"]["STAGE"]
    mario_td = setup_mario(params)
    sut_names,suts = setup_suts(params)
    res = []
    for sut in suts:
        mean,std_dev = eval_single(mario_td,sut,params.getint("TESTING","N_EVAL"))
        res.append((mean,std_dev))
    with open(f"eval_result_{stage}.txt", "w") as fp:
        fp.write(str(sut_names))
        fp.write("\n")
        for r in res:
            fp.write(str(r))
            fp.write("\n")
    
    
def main(params_file):
    params = configparser.ConfigParser()
    params.read(params_file)
    
    stage = params["SETUP"]["STAGE"]
    model_file_name = params["TESTING"]["MODEL"] #f"mario_800_{stage}.dot"
    mario,scheduler = setup_reachability(model_file_name,"win",stage,params)
    
    render = False
    n_ep = 800
    for e in range(n_ep):
        scheduler.reset()
        obs = mario.reset()

        while True:
            action = scheduler.get_input()
            if action is None:
                #print(f"Sampling a random action in {obs}")
                action = random.choice(["right","jump"])
            obs = mario.step(action,render=render)
            reached_state = scheduler.step_to(action, obs)
            if reached_state is None:
                scheduler.step_to_closest(action,obs)
                #print(f"Scheduler undefined at {obs}")
                #break
            if "game_over" in obs or "win" in obs:
                break
            
if __name__ == "__main__":
    
    import sys
    params_file = None
    
    for s in sys.argv:
        if ".ini" in s:
            params_file = s
    if "test" in sys.argv:
        test_main(params_file)
    elif "eval" in sys.argv:
        eval_main(params_file)
    else:
        main(params_file)
