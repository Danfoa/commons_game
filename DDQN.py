import os
import tensorflow as tf

import gym
import time
import numpy as np
np.random.seed(123)
tf.random.set_seed(123)


# Neural Network Model Defined at Here.
class DeepQNet(tf.keras.Model):

    def __init__(self, num_actions, input_shape, kernel_initializer='glorot_uniform', name=None):
        super(DeepQNet, self).__init__(name=name)
        # you can try different kernel initializer
        self.conv1 = tf.keras.layers.Conv2D(6, 3, strides=(1, 1), padding='same', kernel_initializer=kernel_initializer,
                                            input_shape=input_shape, name="conv1")
        self.conv2 = tf.keras.layers.Conv2D(6, 3, strides=(2, 2), padding='same', kernel_initializer=kernel_initializer,
                                            name="conv2")
        self.flat = tf.keras.layers.Flatten(name="flatten")
        self.fc1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer=kernel_initializer, name="dense1")
        self.logits = tf.keras.layers.Dense(num_actions, name='q_values', kernel_initializer=kernel_initializer)

    # forward propagation
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.logits(x)
        return x

    # a* = argmax_a' Q(s, a')
    def action_value(self, obs):
        q_values = self.predict(obs)
        best_action = np.argmax(q_values, axis=-1)
        return best_action if best_action.shape[0] > 1 else best_action[0], q_values[0]


# To test whether the model works
def test_model():
    env = gym.make('CartPole-v0')
    print('num_actions: ', env.action_space.n)
    model = DeepQNet(env.action_space.n)

    obs = env.reset()
    print('obs_shape: ', obs.shape)

    # tensorflow 2.0: no feed_dict or tf.Session() needed at all
    best_action, q_values = model.action_value(obs[None])
    print('res of test model: ', best_action, q_values)  # 0 [ 0.00896799 -0.02111824]


class DDQNAgent:  # Double Deep Q-Network

    def __init__(self, model, target_model, env, obs_shape, buffer_size=200, learning_rate=.0015, epsilon=.1,
                 epsilon_decay=0.995, min_epsilon=.01, gamma=.9, batch_size=8):
        self.model = model
        self.target_model = target_model
        # gradient clip
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=10.0)
        self.model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError())
        self.model.build((None,) + obs_shape)
        self.target_model.build((None,) + obs_shape)
        # print(self.model.summary())

        # parameters
        self.env = env  # gym environment
        self.lr = learning_rate  # learning step
        self.epsilon = epsilon  # e-greedy when exploring
        self.epsilon_decay = epsilon_decay  # epsilon decay rate
        self.min_epsilon = min_epsilon  # minimum epsilon
        self.gamma = gamma  # discount rate
        self.batch_size = batch_size  # batch_size
        self.num_in_buffer = 0  # transition's num in buffer
        self.buffer_size = buffer_size  # replay buffer size

        # replay buffer params [(s, a, r, ns, done), ...]
        self.obs = np.empty((self.buffer_size,) + obs_shape)
        self.actions = np.empty((self.buffer_size), dtype=np.int8)
        self.rewards = np.empty((self.buffer_size), dtype=np.float32)
        self.dones = np.empty((self.buffer_size), dtype=np.bool)
        self.next_states = np.empty((self.buffer_size,) + obs_shape)
        self.next_idx = 0

    def train_step(self):
        idxes = self.sample(self.batch_size)
        s_batch = self.obs[idxes]
        a_batch = self.actions[idxes]
        r_batch = self.rewards[idxes]
        ns_batch = self.next_states[idxes]
        done_batch = self.dones[idxes]
        # Double Q-Learning, decoupling selection and evaluation of the bootstrap action
        # Action Selection with the current DQN model
        best_action_idxes, _ = self.model.action_value(ns_batch)
        target_q = self.get_target_value(ns_batch)  # Target Q values of next step
        # Evaluation with the target DQN model
        target_q = r_batch + self.gamma * target_q[np.arange(target_q.shape[0]), best_action_idxes] * (1 - done_batch)
        # TODO: Change this ineficiency
        target_f = self.model.predict(s_batch)
        for i, val in enumerate(a_batch):
            target_f[i][val] = target_q[i]

        losses = self.model.train_on_batch(s_batch, target_f)

        return losses

    def evaluation(self, env, render=True):
        obs, done, ep_reward = env.reset(), False, 0
        # one episode until done
        while not done:
            action, q_values = self.model.action_value(obs[None])  # Using [None] to extend its dimension (4,) -> (1, 4)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if render:  # visually show
                env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward

    # store transitions into replay butter
    def store_transition(self, obs, action, reward, next_state, done):
        n_idx = self.next_idx % self.buffer_size
        self.obs[n_idx] = obs
        self.actions[n_idx] = action
        self.rewards[n_idx] = reward
        self.next_states[n_idx] = next_state
        self.dones[n_idx] = done
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    # sample n different indexes
    def sample(self, n):
        assert n < self.num_in_buffer
        res = []
        while True:
            num = np.random.randint(0, self.num_in_buffer)
            if num not in res:
                res.append(num)
            if len(res) == n:
                break
        return res

    # e-greedy
    def get_action(self, best_action):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return best_action

    # assign the current network parameters to target network
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_target_value(self, obs):
        return self.target_model.predict(obs)

    def e_decay(self):
        self.epsilon *= self.epsilon_decay

    def reset_replay_buffer(self):
        self.next_idx = 0
        self.num_in_buffer = 0

    def save_policy(self, path):
        os.makedirs(path, exist_ok=True)
        tf.keras.models.save_model(self.model, path, overwrite=True)

    def load_policy(self, path):
        self.model = tf.keras.models.load_model(path)
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipvalue=10.0)
        self.model.compile(optimizer=opt, loss='mse')
        self.target_model = tf.keras.models.load_model(path)

    @staticmethod
    def from_trained_policy(path_to_model, env, obs_shape, buffer_size=200, learning_rate=.0015, epsilon=.1,
                            epsilon_decay=0.995, min_epsilon=.01, gamma=.9, batch_size=8):

        model = tf.keras.models.load_model(path_to_model)
        target_model = tf.keras.models.load_model(path_to_model)

        return DDQNAgent(model, target_model, env, obs_shape, buffer_size=buffer_size, learning_rate=learning_rate,
                         epsilon=epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon, gamma=gamma,
                         batch_size=batch_size)
