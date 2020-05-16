import os
import sys
import glob
import numpy as np
import shutil
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import skimage
import multiprocessing
from multiprocessing import Process, Pool

from tqdm import tqdm
import tensorflow as tf
from social_dilemmas.common_game.commons_env import HarvestCommonsEnv
from social_dilemmas.envs.agent import BASE_ACTIONS, HARVEST_VIEW_SIZE

from DDQN import DDQNAgent, DeepQNet

from joblib import Parallel, delayed
import utility_funcs


def train_agent_on_batch(agent):
    return agent.train_step()


MEDIUM_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P   P   A    A    A    A  A    A    @',
    '@  AAA  AAA  AAA  AAA  AAAAAA  AAA   @',
    '@ A A    A    A    A    A  A    A   P@',
    '@PA                                  @',
    '@ A   A    A    A    A       A    A  @',
    '@PA  AAA  AAA  AAA  AAA     AAA  AAA @',
    '@ A   A    A    A    A   P   A    A  @',
    '@PA                                P @',
    '@ A    A    A    A    A  A    A    A @',
    '@AAA  AAA  AAA  AAA  AA AAA  AAA  AA @',
    '@ A    A    A    A    A  A    A    A @',
    '@P                     P             @',
    '@P  A    A    A    A       P     P   @',
    '@  AAA  AAA  AAA  AAA         P    P @',
    '@P  A    A    A    A   P   P  P  P   @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

SMALL_HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@P A    A    A    A    AP@',
    '@ AAA  AAA  AAA  AAA  AAA@',
    '@  A    A    A    A    A @',
    '@                        @',
    '@    A    A    A    A    @',
    '@   AAA  AAA  AAA  AAA   @',
    '@P   A    A    A    A   P@',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@', ]

MAP = {"small": SMALL_HARVEST_MAP,
       "medium": MEDIUM_HARVEST_MAP}

EPISODES = 1000
STEPS = 1000

LR = 0.0015
gamma = 0.99
epsilon = 0.15
epsilon_dacay = 0.995
N_AGENTS = 1
REPLAY_BUFFER_SIZE = 200
TARGET_UPDATE_ITERATION = 200
EPISODE_RECORD_FREQ = int(EPISODES / 10)
START_LEARNING = 100
BATCH_SIZE = 8
KERNEL_INITIALIZER = 'glorot_uniform'
AGENT_VIEW_RANGE = 5


def train_agents(n_agents=4, map_type="small", logs_path="logs", n_episodes=EPISODES, n_steps=STEPS,
                 batch_size=BATCH_SIZE, lr=0.0015, gamma=0.99, epsilon=0.10, epsilon_dacay=0.995):
    # Configure expermiment logs
    metrics = None
    logdir = logs_path + "/MAP=%s-AGENTS=%d-lr=%.5f-e=%.2f-ed=%.3f-g=%.2f-b=%d" % (map_type, n_agents, lr, epsilon,
                                                                                   epsilon_dacay, gamma, batch_size)
    social_metrics_writer = tf.summary.create_file_writer(logdir + "/social_metrics")

    env = HarvestCommonsEnv(ascii_map=MAP[map_type], num_agents=n_agents, render=True,
                            agent_view_range=AGENT_VIEW_RANGE)
    obs = env.reset()

    # Instanciate DDQN agent models
    ddqn_models = {}
    for agent_id, agent in env.agents.items():
        obs_shape = (2 * AGENT_VIEW_RANGE + 1, 2 * AGENT_VIEW_RANGE + 1, 3)
        q_net_model = DeepQNet(num_actions=env.action_space.n, input_shape=obs_shape,
                               kernel_initializer=KERNEL_INITIALIZER)
        q_net_target = DeepQNet(num_actions=env.action_space.n, input_shape=obs_shape,
                                kernel_initializer=KERNEL_INITIALIZER)
        ddqn_models[agent_id] = DDQNAgent(model=q_net_model, target_model=q_net_target, obs_shape=obs_shape,
                                          env=env, buffer_size=REPLAY_BUFFER_SIZE, learning_rate=lr, epsilon=epsilon,
                                          epsilon_decay=epsilon_dacay, gamma=gamma, batch_size=batch_size)

    for episode in range(n_episodes + 1):
        print("Episode %d" % episode)
        episode_path = logdir + "/n_episodes/episode=%04d" % episode
        models_path = logdir + "/model/episode=%04d" % episode
        if episode % EPISODE_RECORD_FREQ == 0:
            os.makedirs(episode_path, exist_ok=True)

        obs = env.reset()

        # Convert initial observations to [0,1] float imgs
        for agent_id in env.agents.keys():
            # Convert observation image from (Int to float)
            obs[agent_id] = (obs[agent_id] / 255.).astype(np.float32)
            # Reset replay buffers
            ddqn_models[agent_id].reset_replay_buffer()
            ddqn_models[agent_id].e_decay()

        for t in tqdm(range(1, n_steps), desc="Steps", position=0, leave=True, file=sys.stdout):
            # Select agent actions to take
            actions = {}
            for agent_id, agent in env.agents.items():
                # Follow e greedy policy using Q(s,a) function approximator
                best_action, q_values = ddqn_models[agent_id].model.action_value(tf.expand_dims(obs[agent_id], axis=0))
                actions[agent_id] = ddqn_models[agent_id].get_action(best_action)
            # Apply agents actions on the environment
            next_obs, rewards, dones, info = env.step(actions)
            # Store transition in each agent replay buffer
            for agent_id in env.agents.keys():
                next_obs[agent_id] = (next_obs[agent_id] / 255.).astype(np.float32)
                ddqn_models[agent_id].store_transition(obs[agent_id], actions[agent_id], rewards[agent_id],
                                                       next_obs[agent_id], dones[agent_id])
                ddqn_models[agent_id].num_in_buffer = min(ddqn_models[agent_id].num_in_buffer + 1, REPLAY_BUFFER_SIZE)

            # When enough experience is collected, start on-line learning
            if t > START_LEARNING:
                # TODO: Paralelize
                losses = []
                for agent_id in env.agents.keys():
                    loss = ddqn_models[agent_id].train_step()
                    losses.append(loss)

                if t % TARGET_UPDATE_ITERATION == 0:  # Update target model with learned changes
                    for agent_id in env.agents.keys():
                        ddqn_models[agent_id].update_target_model()

                    # with social_metrics_writer.as_default():
                    # tf.summary.histogram('mse_Q', data=losses, step=t + (n_steps * episode))

            # Update current state observations
            obs = next_obs
            # Update the environment social metrics
            env.update_social_metrics(rewards)

            if episode % EPISODE_RECORD_FREQ == 0:
                env.render(episode_path + "/t=%04d.png" % t, title="t=%04d" % t)

        # TODO: Save agent function approximators.

        # Log metrics to tensorboard
        social_metrics = env.get_social_metrics(episode_steps=n_steps)
        efficiency, equality, sustainability, peace = social_metrics
        with social_metrics_writer.as_default():
            tf.summary.scalar('efficiency', data=efficiency, step=episode)
            tf.summary.scalar('equality', data=equality, step=episode)
            tf.summary.scalar('sustainability', data=sustainability, step=episode)
            tf.summary.scalar('peace', data=peace, step=episode)
            # Log agent accumulated reward distribution
            agent_rewards = [np.sum(rewards) for rewards in env.rewards_record.values()]
            tf.summary.histogram('accumulated_reward', agent_rewards, step=episode)

        if metrics is None:
            metrics = np.array(social_metrics)
        else:
            metrics = np.vstack([metrics, np.array(social_metrics)])

        # Make video of episode
        if episode % EPISODE_RECORD_FREQ == 0:
            utility_funcs.make_video_from_image_dir(vid_path=logdir + "/n_episodes",
                                                    img_folder=episode_path,
                                                    video_name="episode=%04d" % episode,
                                                    fps=10)
            # Delete images
            shutil.rmtree(episode_path, ignore_errors=True)
            # Save models Q value NN function approximators
            for agent_id in env.agents.keys():
                ddqn_models[agent_id].save_policy(path=models_path + "/%s" % agent_id)

    # Save metrics as np array for easy plotting
    np.save(logdir + "/social_metrics.npy", metrics)


def gen_episode_video(models_path, map_type, n_agents, video_path):
    os.makedirs(video_path + "/imgs", exist_ok=True)

    STEPS = 1000

    selected_map = MAP[map_type]
    AGENT_VIEW_RANGE = 5

    # Create environment
    env = HarvestCommonsEnv(ascii_map=selected_map, num_agents=n_agents, render=True,
                            agent_view_range=AGENT_VIEW_RANGE)

    # Instanciate DDQN agent models
    ddqn_models = {}
    for agent_id, agent in env.agents.items():
        obs_shape = (2 * AGENT_VIEW_RANGE + 1, 2 * AGENT_VIEW_RANGE + 1, 3)
        model_path = os.path.join(models_path, agent_id)
        ddqn_models[agent_id] = DDQNAgent.from_trained_policy(path_to_model=model_path, obs_shape=obs_shape,
                                                              env=env)

    obs = env.reset()

    # Convert initial observations to [0,1] float imgs
    for agent_id in env.agents.keys():
        # Convert observation image from (Int to float)
        obs[agent_id] = (obs[agent_id] / 255.).astype(np.float32)

    for t in tqdm(range(1, STEPS), desc="Steps", position=0, leave=True):
        # Select agent actions to take
        actions = {}
        for agent_id, agent in env.agents.items():
            # Follow policy using Q(s,a) function approximator
            best_action, q_values = ddqn_models[agent_id].model.action_value(tf.expand_dims(obs[agent_id], axis=0))
            actions[agent_id] = ddqn_models[agent_id].get_action(best_action)

        # Apply agents actions on the environment
        next_obs, rewards, dones, info = env.step(actions)

        # Update current state observations
        obs = next_obs

        # Update the environment social metrics
        env.update_social_metrics(rewards)
        env.render(video_path + "/imgs/t=%04d.png" % t, title="t=%04d" % t)

    # Make video of episode
    utility_funcs.make_video_from_image_dir(vid_path=video_path,
                                            img_folder=video_path + "/imgs",
                                            video_name="learned_policy",
                                            fps=10)
    # Delete images
    shutil.rmtree(video_path + "/imgs", ignore_errors=True)


if __name__ == "__main__":

    params1 = {"n_agents": 4, "map_type": "small", "logs_path": "logs", "n_episodes": EPISODES, "n_steps": STEPS,
               "batch_size": BATCH_SIZE, "lr": 0.0015, "gamma": 0.99, "epsilon": 0.15, "epsilon_dacay": 0.995}

    params2 = {"n_agents": 4, "map_type": "small", "logs_path": "logs", "n_episodes": EPISODES, "n_steps": STEPS,
               "batch_size": BATCH_SIZE, "lr": 0.0005, "gamma": 0.99, "epsilon": 0.15, "epsilon_dacay": 0.999}

    processes = []

    for i, params in enumerate([params1.values(), params2.values()]):
        p = Process(target=train_agents(), args=params, name="Exp%d" % i)
        p.start()
        processes.append(p)
        print(p.name)

    for p in processes:
        p.join()
