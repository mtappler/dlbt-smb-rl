import torch

"""
This file contains some utility functions for fuzzing and search.
"""

# global variable that controls the state abstraction, if True, we consider only x and y coordinate in the abstraction
# otherwise, we include speed and momentum
use_small_state_space = False
run_trace_steps = 0

# replay an existing trace for debugging, e.g., finding a trace leading to the
# flag (such a trace might not exist if we skip too many frames)
def run_trace(unwrapped_env, env, replay_trace, do_render, visited_state_list=None, do_print=False):
    """
    This function runs am action trace in the environment to determine if it is successful and the abstract states
    visited states along the trace. The execution can also be rendered it for debugging purpose.
    Args:
        unwrapped_env: unwrapped env. without transformations
        env: wrapped env. with transformations applied
        replay_trace: the action trace to be executed
        do_render: True if execution shall be rendered, False otherwise
        visited_state_list: ignored if None, otherwise the function expects a list, and it adds the visited states to
        the list
        do_print: if True, the executed actions and info dictionaries return from steps are printed to the console
        (for debugging)

    Returns: pair (success, done) of Booleans where success indicates if the goal is reached by the execution, done
    indicates if a terminal reached is reached

    """
    global run_trace_steps
    env.reset()
    success = False
    for i, a in enumerate(replay_trace):
        # perform action in environment
        next_rl_state, reward, done, info = env.step(a)
        run_trace_steps += 1
        if do_render:
            env.render()
        # keep track of visited abstract states
        if visited_state_list is not None:
            visited_state_list.append(make_search_state(unwrapped_env, info, None))
        # print info to console
        if do_print:
            print(f"Action {i}: {a}")
            print(info)
            print(done)
        if done:
            if info['flag_get']:
                success = True
            break
    return success, done


