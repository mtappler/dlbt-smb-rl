import numpy as np
import time, datetime
import matplotlib.pyplot as plt

class MetricLogger:
    """
    A utility for logging statistics of learning, such as, mean rewards, mean Q values, times, loss values, and
    exploration rate.
    """
    def __init__(self, save_dir):
        """
        Constructors to initialize the logger, by writing the header log file for training and setting file names
        for the images where we plot data. Additionally, we initialize all metrics that we collect with empty lists.
        Args:
            save_dir: the directory where we save the logs
        """
        self.save_log = save_dir / "log"
        # write the log header
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        # set file names
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()


    def log_step(self, reward, loss, q):
        """
        Record a single step, i.e., executing an action in the environment for training
        Args:
            reward: reward gained in the step
            loss: loss from learning, None if there was no learning update in the step
            q: average q value from learning, None if there was no learning update in the step

        Returns: None

        """
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """
        With this function, we mark the end of an episodes, i.e., we store the rewards gained in the episode and
        initialize a new lists of rewards to collect more of them in another episode. We also compute averages for the
        episode.
        Returns: None

        """
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            # compute averages
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        # append the averages to the list of averages
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        """
        We initialize all the metrics that we log to zero
        Returns: None

        """
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        """
        Record statistics from episodes and the exploration rate. The statistics logged are moving averages from
        multiple episodes, which we print to the log file in regular intervals.
        Args:
            episode: the number of episodes so far
            epsilon: the current exploration rate
            step: the number steps performed so far for training

        Returns: None

        """
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)


        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        # print episode data to the console
        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        # append the data to the log file
        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )
        # add the data to the plot images as well
        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()


class EvaluationLogger:
    """
    Another logger class for logging evaluation data, i.e., the data from evaluating intermediate policies in the
    environment.
    """
    def __init__(self, save_dir, eval_mode = False):
        """
        Constructor that initializes everything and create an eval log file.
        Args:
            save_dir: the directory where the logs shall be stored
            eval_mode: true if we are in eval mode, i.e., if we evaluate an existing agent, rather than evaluating
            an agent that is currently being trained
        """
        self.save_log = save_dir / ("eval" if not eval_mode else "eval_mode")
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episodes':>8}{'Steps':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MedianReward':>15}{'MaxReward':>15}{'MeanLength':>15}{'MinReward':>15}{'AllRewards':>15}\n"
            )

        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.ep_rewards = []
        self.ep_lengths = []

    def init_episode(self):
        """
        Initialize the data recorded for the current episode, i.e., reward and episode length.

        Returns: None

        """
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0

    def log_step(self, reward):
        """
        Log data from a single steps, i.e., just the reward.
        Args:
            reward: gained reward by the executed action

        Returns: None

        """
        self.curr_ep_reward += reward
        self.curr_ep_length += 1

    def log_episode(self):
        """
        End the episode, i.e., store the list of rewards gained in the episode and the length of the episodes.

        Returns: None

        """
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        # reinitialize for a new episode
        self.init_episode()

    def log_evaluation_cycle(self, mario, old_exp_rate):
        """
        Log all the data from the current evaluation cycle, i.e., statistics of the rewards gained, like mean and
        median over several evaluation episodes.
        Args:
            mario: the RL agent
            old_exp_rate: the exploration rate (relevant if we evaluate during training)

        Returns:

        """
        # compute statistics
        mean_ep_reward = np.round(np.mean(self.ep_rewards), 3)
        median_ep_reward = np.round(np.median(self.ep_rewards), 3)
        max_ep_reward = np.round(np.max(self.ep_rewards), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths), 3)
        min_ep_reward = np.round(np.max(self.ep_lengths), 3)
        rewards_copy = list(self.ep_rewards)

        # reinitialize
        self.ep_rewards.clear()
        self.ep_lengths.clear()

        # append to evaluation log file
        with open(self.save_log, "a") as f:
            f.write(
                f"{mario.curr_episode:9d}{mario.curr_step:9d}{old_exp_rate:10.3f}"
                f"{mean_ep_reward:15.3f}{median_ep_reward:15.3f}{max_ep_reward:15.3f}{mean_ep_length:15.3f}"
                f"{min_ep_reward:15.3f} {rewards_copy}\n"
            )
