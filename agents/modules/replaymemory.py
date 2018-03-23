import numpy as np


############################################################

def print_progress(msg):
    """
    Print progress on a single line and overwrite the line.
    Used during optimization.
    """

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()

############################################################

class ReplayMemory:
    """
    The replay-memory holds many previous states of the game-environment.
    This helps stabilize training of the Neural Network because the data
    is more diverse when sampled over thousands of different states.
    """

    def __init__(self, state_shape, size, num_actions, discount_factor=0.97):

        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)
        self.actions = np.zeros(shape=size, dtype=np.int)
        self.rewards = np.zeros(shape=size, dtype=np.float)
        self.end_life = np.zeros(shape=size, dtype=np.bool)
        self.end_episode = np.zeros(shape=size, dtype=np.bool)

        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        self.size = size
        self.discount_factor = discount_factor
        self.num_used = 0
        self.error_threshold = 0.1

    def is_full(self):
        return self.num_used == self.size

    def used_fraction(self):
        return self.num_used / self.size

    def reset(self):
        self.num_used = 0

    def add(self, state, q_values, action, reward, end_life, end_episode):
        if not self.is_full():
            k = self.num_used
            self.num_used += 1
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_life[k] = end_life
            self.end_episode[k] = end_episode

            # The reward is limited to stabilize the training
            self.rewards[k] = np.clip(reward, -1.0, 1.0)

    def update_all_q_values(self):
        # TODO

    def prepare_sampling_prob(self, batch_size=128):
        # TODO

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.
        
        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """
        idx_lo = np.random.choice(self.idx_err_lo, size=self.num_samples_err_lo, replace=False)

        idx_hi = np.random.choice(self.idx_err_hi, size=self.num_samples_err_hi, replace=False)

        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]
        return states_batch, q_values_batch

    def all_batches(self, batch_size=128):
        """
        A generator, iterate to get batch orderly from the replay memory buffer.
        """
        begin = 0
        while begin < self.num_used:
            end = begin + batch_size
            if end > self.num_used:
                end = self.num_used
            progress = end / self.num_used

            yield begin, end, progress

            begin = end

    def estimate_all_values(self, model):
        """
        Estimate all Q-values for the states in the replay-memory
        using the model / Neural Network.

        Note that this function is not currently being used. It is provided
        to make it easier for you to experiment with this code, by showing
        you an efficient way to iterate over all the states and Q-values.

        :param model:
            Instance of the NeuralNetwork-class.
        """
        # TODO the input model need revise

        print("Re-calculating all Q-values in replay memory ...")
         # Process the entire replay-memory in batches.
        for begin, end, progress in self.all_batches():
            # Print progress.
            msg = "\tProgress: {0:.0%}"
            msg = msg.format(progress)
            print_progress(msg)

            # Get the states for the current batch.
            states = self.states[begin:end]

            # Calculate the Q-values using the Neural Network
            # and update the replay-memory.
            self.q_values[begin:end] = model.get_q_values(states=states)

        # Newline.
        print()

    def print_statistics(self):
        """Print statistics for the contents of the replay-memory."""

        print("Replay-memory statistics:")

        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values_old),
                         np.mean(self.q_values_old),
                         np.max(self.q_values_old)))

        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values),
                         np.mean(self.q_values),
                         np.max(self.q_values)))

        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(q_dif),
                         np.mean(q_dif),
                         np.max(q_dif)))

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        print(msg.format(self.error_threshold, err_count,
                         self.num_used, err_count / self.num_used))

        # How much of the replay-memory is used by states with end_life.
        end_life_pct = np.count_nonzero(self.end_life) / self.num_used

        # How much of the replay-memory is used by states with end_episode.
        end_episode_pct = np.count_nonzero(self.end_episode) / self.num_used

        # How much of the replay-memory is used by states with non-zero reward.
        reward_nonzero_pct = np.count_nonzero(self.rewards) / self.num_used

        # Print those statistics.
        msg = "\tend_life: {0:.1%}, end_episode: {1:.1%}, reward non-zero: {2:.1%}"
        print(msg.format(end_life_pct, end_episode_pct, reward_nonzero_pct))
