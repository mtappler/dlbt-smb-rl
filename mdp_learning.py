import os
import datetime
import logging
import configparser
import pickle
import time
from pathlib import Path

from aalpy.learning_algs import run_JAlergia
import gym_super_mario_bros
import torch
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from gym_super_mario_bros import actions
from nes_py.wrappers import JoypadSpace
import numpy as np

import util
from fuzzing import fuzz
from search import search

import random
from metrics import MetricLogger, EvaluationLogger
from agent import Mario
from wrappers import ResizeObservation, SkipFrame
from util import run_trace_steps
from aalpy.utils import load_automaton_from_file, mdp_2_prism_format
from gym import Wrapper

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

log = logging.getLogger("FooBar")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s | %(levelname)-10s | %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)
log.addHandler(handler)
MAX_PRETRAIN_TRACES = 150

params = configparser.ConfigParser()
eval_logger = None

def setup(check_point):
    global params
    
    env = gym_super_mario_bros.make(f"SuperMarioBros-{params.get('SETUP', 'STAGE')}-{params.get('SETUP', 'STYLE')}")
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
        
    # directory where neural networks and intermediate results and data are stored
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, params=params,
                  checkpoint=check_point,load_only_conv=False,disable_cuda=False)

    env.reset()
    return env, unwrapped_env, mario

def collect_samples(env, unwrapped, mario, n_samples, max_x = 10000):
    act_map = {0 : "right", 1: "jump"}
    samples = []
    
    while (e := mario.curr_episode) <= n_samples:
        log.debug(f"Running episode {e}")

        # reset the environment
        state = env.reset()
        state = torch.from_numpy(np.array(state)).float()

        episode = []
        # Play the game!
        sample = []
        sample.append("Init")
        while True:
            # Pick an action
            action = mario.act(state,eval_mode=True)

            # Perform action
            next_state, reward, done, info = env.step(action)
            next_state = torch.from_numpy(np.array(next_state)).float()
            obs = [f"pos_{info['x_pos']}_{info['y_pos']}"]

            # Update state
            state = next_state

            # Check if end of game
            if done or info['flag_get']:
                if info['flag_get']:
                    print("Win")
                    obs.append("win")
                else:
                    obs.append("game_over")
            sample.extend((act_map[action], "__".join(obs)))
            #if done or len(sample) > 300:
            #    if info['x_pos'] > max_x:
            #        mario.curr_episode += 1
            #        samples.append(sample)
            #        print(f"We have {mario.curr_episode}")
            #        print(sample)
            #    else:
            #        print(info['x_pos'])
            #    break
            if done or info['x_pos'] > max_x:
                mario.curr_episode += 1
                samples.append(sample)
                print(sample)
                break
    return samples


def main(eval_mode = False, params_file = None):
    """
    Main method for the whole DQfD with fuzzed demonstrations and for plain DDQ.

    Args:
        eval_mode: Boolean value indicating whether a saved agent shall just be evaluated rather than trained
        params_file: path to an ini-file containing the configuration for learning

    Returns: None

    """
    global params

    if params_file:
        params.read(params_file)
    else:
        params.read('params.ini')

    stage = params.get('SETUP', 'STAGE')
    checkpoint_path = params.get('TRAINING', 'CHECKPOINT')
    checkpoint = Path(checkpoint_path) if checkpoint_path != 'None' else None
    
    env, unwrapped_env,mario = setup(checkpoint)
    n_samples = 800
    if "MAX_X_POS" in params["TRAINING"]:
        max_x = params.getint("TRAINING","MAX_X_POS")
    else:
        max_x = 10000
    samples = collect_samples(env, unwrapped_env, mario, n_samples=n_samples, max_x = max_x)
    
    model = run_JAlergia(samples, automaton_type='mdp', path_to_jAlergia_jar='alergia.jar', heap_memory='-Xmx12G',
                         optimize_for='memory')
    model.save(f'mario_{n_samples}_{stage}.dot')
  
    mdp_2_prism_format(model, "mario", f"mario_{n_samples}_{stage}.prism")
    
if __name__ == '__main__':
    """
    Main of the python file, which expects an ini-file for configuration
    """
    import sys
    params_file = None
    sys. setrecursionlimit(5000)
    for s in sys.argv:
        if ".ini" in s:
            params_file = s
        
    
    main(eval_mode = False, params_file = params_file)
