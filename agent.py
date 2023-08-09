import logging
import pickle
from pathlib import Path

import torch
import random
import numpy as np

from neural import MarioNet
from collections import deque

from torch.autograd import Variable

log = logging.getLogger("FooBar")


class Mario:
    """
    The RL agent, this class contains the logic for caching experiences, both normal and expert, as well as for learning
    i.e., updating the policy network.
    """

    def __init__(self, state_dim, action_dim, save_dir, params, checkpoint=None, load_only_conv=False, disable_cuda=False):
        """

        Args:
            state_dim: the dimension of states, i.e., the size of images and how many of them are stacked
            action_dim: number of actions
            save_dir: directory where policy networks shall be stored
            params: configuration parameters
            checkpoint: pretrained neural network (default = None)
            load_only_conv: Boolean value indicating whether only the convolutional part of the policy network shall be
            loaded while the rest is initialized to default values
        """
        global log

        log.debug("Initializing Mario Agent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        # normal buffer is a fixed double-ended queue
        self.memory = deque(maxlen=params.getint('TRAINING', 'MEMORY_SIZE'))
        # Expert memory can grow indefinitely (limit is set in main
        self.expert_memory = deque()
        self.use_cuda = torch.cuda.is_available() if not disable_cuda else False
        self.device = 'cuda' if self.use_cuda else 'cpu'
        # size of minibatches
        self.batch_size = params.getint('TRAINING', 'BATCH_SIZE')

        self.pretrain_steps = params.getint('TRAINING', 'PRETRAIN_STEPS')
        self.expert_knowledge_share = 0 if self.pretrain_steps == 0 else 0.5
        # CHANGED from 0.25
        self.cache_device = 'cpu' if params.getint('TRAINING', 'MEMORY_SIZE') > 5000 else self.device 
        self.expert_recall_size = int(self.batch_size * self.expert_knowledge_share)
        self.margin = params.getfloat('TRAINING', 'MARGIN')

        self.exploration_rate = params.getfloat('TRAINING', 'EXPLORATION_RATE_INIT')
        self.exploration_rate_decay = params.getfloat('TRAINING', 'EXPLORATION_RATE_DECAY')
        self.exploration_rate_min = params.getfloat('TRAINING', 'EXPLORATION_RATE_MIN')
        self.gamma = params.getfloat('TRAINING', 'GAMMA')

        self.curr_step = 0
        self.curr_pretrain_step = 0
        self.curr_episode = 0
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 10000
        # how often intermediate policies are evaluated
        self.eval_every = params.getint('LOGGING', 'EVAL_EVERY')

        # how often intermediate policies are dumped to a file
        self.save_every = params.getint('LOGGING', 'SAVE_EVERY')
        self.save_dir = save_dir
        self.evaluation_randomness = self.exploration_rate_min
        self.expert_cache_save = []
        
        log.debug(f"Cuda available: {self.use_cuda}")

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

        if checkpoint:
            log.debug(f"Loading previous checkpoint: {checkpoint}")
            self.load(checkpoint,load_only_conv)

        # we use the Adam optimizer for learning
        # lowered lr from 0.00025 to 0.00005
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, weight_decay = 5e-5) # added weight_decay
        # the loss function for standard TD updates
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        # the loss function for the n-step updates
        self.n_step_loss_fn = torch.nn.SmoothL1Loss(reduction='mean')
        log.debug("Mario Agent initialized")

    def act(self, state, eval_mode=False):
        """
        Action selection for interacting with the environment. Depending on the current exploration rate, either a
        random action will be selected or one predicted by the policy network.
        Args:
            state: current state of the environment
            eval_mode: Boolean value indicating whether we currently evaluate a policy where the probability of a
            random selection is very low

        Returns: an action

        """
        # EXPLORE (choose a random action)
        if np.random.rand() < (self.exploration_rate if not eval_mode else self.evaluation_randomness):
            action_idx = np.random.randint(self.action_dim)
        # EXPLOIT (choose action with the policy network)
        else:
            state = torch.from_numpy(np.array(state)).float().to(self.device)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()

        if eval_mode:
            return action_idx

        # decrease exploration_rate (eval does not decrease exploration)
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1

        return action_idx
    
    def compute_n_step_return(self,n_rewards, device):
        """
        Computation of the n-step returns, i.e., discounted rewards accumulated over n steps
        Args:
            n_rewards: rewards of n steps and the last state reached after the steps
            device: GPU or CPU

        Returns: triple of summed and discounted rewards, discount for the Q-value of the reached state, and the reached
        state on the chosen device

        """
        # n_step_return
        (rewards, last_state) = n_rewards
        # initialize
        n_step_return = 0
        # there is probably a more efficient, batched way to do this, maybe like that, but we do the naive implementation
        # for now
        #last_state_Q = self.net(last_state.to(self.device).unsqueeze(0).repeat(self.batch_size,1,1,1), model='target')[0]
        #last_Q_max = torch.max(last_state_Q, axis=0)[0]
        #n_step_return = last_Q_max

        # sum discounted rewards
        for r in reversed(rewards):
            n_step_return *= self.gamma
            n_step_return += r
        # compute discount for the Q-value of the last state
        last_Q_discount = self.gamma ** len(rewards)
        if last_state is None:
            # if we reached a terminal, it does not factor into the computation of the return
            last_Q_discount = 0.0
            last_state = torch.zeros(self.state_dim).to(device).float()

        # send to device
        n_step_return = torch.tensor(n_step_return).to(device).float()
        last_Q_discount = torch.tensor(last_Q_discount).to(device).float()
        # n_step_return end
        return (n_step_return, last_Q_discount, last_state.to(device))

    def refresh_expert_cache(self):
        """
        Reinitialize cached computations for expert, not used in experiments. Only used in experimental to check for
        bugs.
        Returns: None

        """
        print("Refreshing expert cache")
        self.expert_memory = deque()
        for (state, next_state, action, reward, n_rewards, done) in self.expert_cache_save:
            self._cache_expert(state, next_state, action, reward,n_rewards, done)

    def cache_expert(self, state, next_state, action, reward,n_rewards, done):
        """
        Fill expert cache with experiences
        Args:
            state: start state
            next_state: next state
            action: performed action
            reward: gained rewards
            n_rewards: a pair of n rewards of subsequent steps (or less if terminal state is reached) and state reached
            after n steps
            done: Boolean indicating whether a terminal state was reached

        Returns: None

        """
        self.expert_cache_save.append((state, next_state, action, reward, n_rewards, done,))
        self._cache_expert(state, next_state, action, reward,n_rewards, done)

    def _cache_expert(self, state, next_state, action, reward,n_rewards, done):
        """
        Internal function that fills the expert cache. It transforms all parts of experience into tensors and sends
        them to the CPU. We use the CPU, because the expert cache does not have a fixed size, so the GPU RAM may not be
        sufficient.
        Args:
            state: same as cache_expert
            next_state: same as cache_expert
            action: same as cache_expert
            reward: same as cache_expert
            n_rewards: same as cache_expert
            done: same as cache_expert

        Returns: None

        """
        state = state.to('cpu').float()
        next_state = next_state.to('cpu').float()
        action = torch.tensor(action).to('cpu').long()
        reward = torch.tensor(reward).to('cpu').double()
        n_step_return = self.compute_n_step_return(n_rewards, 'cpu')
        done = torch.tensor(done).to('cpu').bool()
        self.expert_memory.append((state, next_state, action, reward, n_step_return, done,))
    
    def cache(self, state, next_state, action, reward,n_rewards,done):
        """
        Function that fills the normal cache. It transforms all parts of experience into tensors and sends
        them to the chosen device. We may use the GPU for the fixed-size normal cache.
        Args:
            state: start state
            next_state: next state
            action: performed action
            reward: gained rewards
            n_rewards: a pair of n rewards of subsequent steps (or less if terminal state is reached) and state reached
            after n steps
            done: Boolean indicating whether a terminal state was reached

        Returns: None

        """
        state = state.to(self.cache_device).float()
        next_state = next_state.to(self.cache_device).float()
        action = torch.tensor(action).to(self.cache_device).long()
        reward = torch.tensor(reward).to(self.cache_device).double()
        n_step_return = self.compute_n_step_return(n_rewards, self.cache_device)
        done = torch.tensor(done).to(self.cache_device).bool()
        self.memory.append((state, next_state, action, reward,n_step_return, done,))

    def recall(self, bs=None):
        """
        Sample experiences from the normal cache, aggregate them for performing minibatch updates and send them to the
        device chosen for computation, i.e., CPU or GPU.
        Args:
            bs: batch size, if None we use the batch size configured in the constructor

        Returns: a sample of experiences

        """
        if bs is None:
            replay_batch = random.sample(self.memory, self.batch_size)
        else:
            replay_batch = random.sample(self.memory, bs)

        #state, next_state, action, reward,n_step_return, done = map(torch.stack, zip(*replay_batch))
        state, next_state, action, reward,n_step_return_info, done = zip(*replay_batch)
        n_step_return, last_Q_discount, last_state = zip(*n_step_return_info)
         
        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        n_step_return, last_Q_discount, last_state = map(torch.stack, [n_step_return, last_Q_discount, last_state])
        if self.device != self.cache_device:
            return state.to(self.device), next_state.to(self.device), action.to(self.device), reward.to(self.device),(n_step_return.to(self.device), last_Q_discount.to(self.device), last_state.to(self.device)),done.to(self.device)
        else:
            return state, next_state, action, reward,(n_step_return, last_Q_discount, last_state), done

    def expert_recall(self, bs=None):
        """
        Sample experiences from the expert cache, aggregate them for performing minibatch updates and send them to the
        device chosen for computation, i.e., CPU or GPU.
        Args:
            bs: batch size, if None we use the batch size configured in the constructor

        Returns: a sample of experiences

        """
        if bs is None:
            replay_batch = random.sample(self.expert_memory, self.batch_size)
        else:
            replay_batch = random.sample(self.expert_memory, bs)

        state, next_state, action, reward,n_step_return_info, done = zip(*replay_batch)
        n_step_return, last_Q_discount, last_state = zip(*n_step_return_info)
         
        state, next_state, action, reward, done = map(torch.stack, [state, next_state, action, reward, done])
        n_step_return, last_Q_discount, last_state = map(torch.stack, [n_step_return, last_Q_discount, last_state])
            
        #state, next_state, action, reward,n_step_return, done = map(torch.stack, zip(*replay_batch))
        return state.to(self.device), next_state.to(self.device), action.to(self.device), reward.to(self.device),(n_step_return.to(self.device), last_Q_discount.to(self.device), last_state.to(self.device)),done.to(self.device)

    def td_estimate(self, state, action):
        """
        TD estimate from the online network, i.e, Q value for state-action pair from Q-online network.
        Args:
            state: a state for the estimate
            action: an action for the estimate

        Returns: pair of the Q-values for the state and the Q-value for the state-action pair

        """
        state = self.net(state, model='online')
        state_action = state[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return state, state_action

    def q_online(self, states, actions):
        """
        Function for getting Q-values from the online network for multiple state-action pairs.
        Args:
            states: an array of states
            actions: an array of actions

        Returns: an array of Q-values for the state-action pairs

        """
        state_Q = self.net(states, model='online')[np.arange(0, self.batch_size), actions]
        return state_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        TD value from the target network, i.e, maximum Q value for a state from the Q-target network multiplied
        by the discount factor and with added reward.
        Args:
            reward: reward gained in a step
            next_state: state reached by a step
            done: indicating whether a terminal state was reached (if so, we return only the reward)

        Returns: maximum Q-value in the reached state

        """
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def supervised_loss(self, q_state, q_state_action, actions, pt=False):
        """
        The large margin classification loss between actions in the demonstrations and actions chosen by the agent,
        i.e., the loss that makes actions in the demonstrations appear better. This loss is only applied to state-action
        pairs from demonstration experiences.
        Args:
            q_state: array of Q-values for states
            q_state_action: array of Q-values for state-action pairs
            actions: actions in demonstrations
            pt: Boolean indicating if we are in the pretraining phase

        Returns:

        """
        # Implementation approach taken from https://github.com/nabergh/doom_dqfd

        # if we are in pretraining we apply the loss to all elements of the minibatch, otherwise only to those
        # corresponding to expert experiences
        ex_s = self.batch_size if pt else self.expert_recall_size

        # helper matrix for with the constant loss value
        margins = (torch.ones(self.action_dim, self.action_dim) -
                   torch.eye(self.action_dim)) * self.margin

        state_margins = q_state + margins.to(self.device)[actions]
        
        #supervised_loss = (state_margins.max(1)[0].unsqueeze(1) - q_state_action).pow(2)[:ex_s].mean()
        supervised_loss = (state_margins.max(1)[0].unsqueeze(1) - q_state_action).abs()[:ex_s].mean()
        return supervised_loss

    @torch.no_grad()
    def n_step_Q(self, last_state):
        """
        Compute the Q-value from the online network for states reached after n-steps to compute the n-step return loss.
        Args:
            last_state: the reached state

        Returns: max Q-value from online network for the given reached states

        """
        last_state_Q = self.net(last_state, model='online')
        best_action = torch.argmax(last_state_Q, axis=1)
        last_Q = self.net(last_state, model='target')[np.arange(0, self.batch_size), best_action]
        return last_Q


    def n_step_loss(self,td_est, states,actions,n_step_info,pt):
        """
        Compute the n-step loss similar to the one-step return loss.
        Args:
            td_est: one-step estimate from online network
            states: unused
            actions: unused
            n_step_info: triple with the data required for computing n-step return
            pt: Boolean indicating whether we are pretraining (unused)

        Returns: n-step return loss

        """
        ex_s = self.batch_size
        n_step_return, last_Q_discount, last_state = n_step_info
        # compute Q-values for states reached after n steps
        last_Q = self.n_step_Q(last_state)
        # compute finally the n-step return
        n_step_return = n_step_return + last_Q * last_Q_discount
        # compute loss against one-step return
        return self.n_step_loss_fn(n_step_return[:ex_s],td_est[:ex_s])        
    
    def update_Q_online(self, td_estimate, td_target, q_states, states,actions, n_step_info, pt=False):
        """
        Perform the actual minibatch update of the online network by aggregating all the losses and performing a step
        with the optimizer.
        Args:
            td_estimate: TD estimate of the online network
            td_target: max Q-values in next states (Q-target network)
            q_states: array of Q-values for states
            states: array of states
            actions: array of actions
            n_step_info: arrays of triples containing information to compute n-step return loss
            pt: Boolean indicating whether we are pretraining

        Returns: sum of all losses

        """
        # trying the following line
        #states.requires_grad = True 
        
        dqn_loss = self.loss_fn(td_estimate, td_target)
        # weights for losses, constantly set to one
        l1 = 1
        l2 = 1
        ns_loss = self.n_step_loss(td_estimate,states,actions,n_step_info,pt)
        loss = dqn_loss + l1 * ns_loss

        # if we have expert demonstrations (i.e., DQfD) in the batch, we compute the supervised loss
        if self.expert_recall_size > 0:
            sup_loss = self.supervised_loss(q_states, td_estimate, actions, pt)
            loss += l2*sup_loss

        # perform the actual weight updates with the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """
        Synchronize the online and the target networks, i.e., copying the weights from the online network over to the
        target network.
        Returns: None

        """
        self.net.target_conv.load_state_dict(self.net.online_conv.state_dict())
        self.net.target_linear.load_state_dict(self.net.online_linear.state_dict())

    def learn(self):
        """
        API function for learning. When called this function will check whether an update shall be performed depending
        on the chosen update intervals of the online network. It will check whether the target and the online shall be
        synchronized. It performs those actions in the configured intervals, so the function may do nothing in between
        intervals.

        Returns:

        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if len(self.memory) < self.batch_size:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Sample from memory
        state, next_state, action, reward, ns_return, done = self.mixed_recall()

        # Get TD Estimate
        q_states, td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt, q_states,state, action, ns_return)

        return td_est.mean().item(), loss

    def save(self, params):
        """
        Save the complete policy network (online and target) to a pickle file as a checkpoint.
        Args:
            params: configuration parameter that are saved as an ini file

        Returns: None

        """
        save_path = self.save_dir / f"mario_net_{int(self.curr_episode)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        with open(self.save_dir / "params.ini","w") as f:
            params.write(f)
        print(f"MarioNet saved to {save_path} at step {self.curr_episode}")

    def load(self, load_path,load_only_conv=False):
        """
        Loads a previously saved policy network to initialize an agent.
        Args:
            load_path: path to the checkpoint
            load_only_conv: Boolean value indicating that only the convolutional shall be loaded and the linear part of
            the network shall be left to their default values

        Returns: None

        """
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)

        log.info(f"Loading previously saved model at {load_path}")
        self.net.load_state_dict(ckp.get('model'))
        self.exploration_rate = ckp.get('exploration_rate')
        # we reset the linear part after loading to get to the default values
        # exploration rate is also reset in this case
        if load_only_conv:
            self.net.reset_linear(self.use_cuda)
            self.exploration_rate = params.getfloat('TRAINING', 'EXPLORATION_RATE_INIT')
        log.info(f"Successfully loaded model")

    def get_gpu_memory(self):
        """
        Helper function for debugging to determine how much GPU memory is available and used.
        Returns: A string indicating memory allocation

        """
        if self.use_cuda:
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            return f'Total memory: {t / 1e9}GB, reserved: {r / 1e9}GB, allocated: {a / 1e9}GB, free: {(r - a) / 1e9}GB \n'
        return 'Cuda not used \n'

    def compute_added_reward(self, info, reward, coin=False, score=False):
        """
        Function to add additional rewards to experiment with other/sparse rewards (not used by default in experiments).
        Args:
            info: info dictionary from the gym environment
            reward: the normal reward
            coin: Boolean whether reward for coins shall be included
            score: Boolean whether reward for score shall be included

        Returns:

        """
        # add 5 points per coin
        # add 15/1000 points per score increase
        current_score = info["score"]
        current_coins = info["coins"]
        # add to normal reward
        if coin:
            reward += 5 * (current_coins - self.previous_coins)
        if score:
            reward += (15 / 1000) * (current_score - self.previous_score)
        self.previous_coins = current_coins
        self.previous_score = current_score
        # clip reward
        reward = min(15, reward)
        return reward

    def dump_expert_memory(self,params):
        """
        Store the expert memory to a pickle file.
        Args:
            params: configuration parameters

        Returns: None

        """
        fuzz_load_path = params.get("MODE_FUZZ","LOAD_PATH")
        save_path = Path(fuzz_load_path.replace('.traces','_init_exp.memory'))

        with open(save_path, 'wb') as file:
            pickle.dump((self.expert_memory,self.expert_cache_save), file)

    def load_expert_memory(self,params):
        """
        Load previously saved expert memory from a pickle file.
        Args:
            params: configuration parameters

        Returns: True if saved memory exists, False otherwise

        """
        fuzz_load_path = params.get("MODE_FUZZ","LOAD_PATH")
        save_path = Path(fuzz_load_path.replace('.traces','_init_exp.memory'))
        if save_path.exists():
            with open(save_path, 'rb') as file:
                (self.expert_memory,self.expert_cache_save) = pickle.load(file)
                return True
        return False

    def pretrain(self):
        """
        Pretraining function for DQfD. This works like the normal learning except that only experiences from the expert
        buffer are used.
        Returns: None

        """
        if self.curr_pretrain_step % 100 == 0:
            log.debug(f"Pretrain step {self.curr_pretrain_step}")

        if self.curr_pretrain_step % self.sync_every == 0:
            self.sync_Q_target()

        # sample experiences only from expert buffer
        state, next_state, action, reward, ns_return, done = self.expert_recall()
        # Get TD Estimate
        q_states, td_est = self.td_estimate(state, action)
        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        self.update_Q_online(td_est, td_tgt, q_states, state, action, ns_return, True)

        td_est.mean().item()

        self.curr_pretrain_step += 1

    def mixed_recall(self):
        """
        Mixed sampling from both expert replay buffer and normal replay buffer, i.e., we sample self.expert_recall_size
        many experiences from the expert buffer and the remaining experiences from the normal buffer to get to
        self.batch_size many experiences.
        If we perform standard DDQ, we sample only from the normal buffer.
        Returns: sample of experiences

        """
        expert_batch_size = self.expert_recall_size
        mario_batch_size = self.batch_size - expert_batch_size

        if expert_batch_size > 0:
            # sample expert experiences
            ex_state, ex_next_state, ex_action, ex_reward, ex_ns_return, ex_done = self.expert_recall(expert_batch_size)
            # sample normal experiences
            state, next_state, action, reward,ns_return, done = self.recall(mario_batch_size)

            ex_n_step_return, ex_last_Q_discount, ex_last_state = ex_ns_return
            n_step_return, last_Q_discount, last_state = ns_return
        
            # concatenate experiences from both buffers so that we first have expert data and then normal data
            r_state = torch.cat((ex_state, state))
            r_next_state = torch.cat((ex_next_state, next_state))
            r_action = torch.cat((ex_action, action))
            r_reward = torch.cat((ex_reward, reward))
            #r_ns_return = torch.cat((ex_ns_return, ns_return))
            r_n_step_return = torch.cat((ex_n_step_return,n_step_return))
            r_last_Q_discount = torch.cat((ex_last_Q_discount,last_Q_discount))
            r_last_state = torch.cat((ex_last_state,last_state))
            #
            r_done = torch.cat((ex_done, done))

            return r_state, r_next_state, r_action, r_reward, (r_n_step_return,r_last_Q_discount,r_last_state), r_done

        else:
            # in standard DDQ we just sample from the normal buffer
            return self.recall()



