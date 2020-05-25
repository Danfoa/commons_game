# Standard library imports
import os
import sys
import shutil
import time
import multiprocessing
from multiprocessing import Process, Pool

import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Local application imports
from game_environment.commons_env import HarvestCommonsEnv, DEFAULT_COLORMAP, MAP
from game_environment.utils import utility_funcs
from DDQN import DDQNAgent, DeepQNet
from game_environment.map_env import DEFAULT_COLOURS



EPISODES = 1000
STEPS = 1000
BATCH_SIZE = 8

LR = 0.0015
gamma = 0.99
epsilon = 0.15
epsilon_dacay = 0.995
N_AGENTS = 1
REPLAY_BUFFER_SIZE = 200
TARGET_UPDATE_ITERATION = 20
EPISODE_RECORD_FREQ = 10
START_LEARNING = 100
KERNEL_INITIALIZER = 'glorot_uniform'
AGENT_VIEW_RANGE = 5


def train_agents(n_agents=4, map_type="small", logs_path="logs", n_episodes=EPISODES, n_steps=STEPS,
                 batch_size=BATCH_SIZE, lr=0.0015, gamma=0.99, epsilon=0.10, epsilon_decay=0.995, log=True):

    # Configure expermiment logs
    logdir = logs_path + "/MAP=%s-AGENTS=%d-lr=%.5f-e=%.2f-ed=%.3f-g=%.2f-b=%d" % (map_type, n_agents, lr, epsilon,
                                                                                   epsilon_decay, gamma, batch_size)
    os.makedirs(logdir, exist_ok=True)
    # sys.stdout = open(os.path.join(logdir, "console_output.out"), "w+")

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
                                          epsilon_decay=epsilon_decay, gamma=gamma, batch_size=batch_size)

    for episode in range(n_episodes + 1):
        start_t = time.time()
        print("- A:%d Episode %d" % (n_agents, episode))
        episode_path = logdir + "/episodes/episode=%04d" % episode
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

        for t in tqdm(range(1, n_steps), position=0, leave=True):
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

        # Make video of episode
        if episode % EPISODE_RECORD_FREQ == 0:
            utility_funcs.make_video_from_image_dir(vid_path=logdir + "/episodes",
                                                    img_folder=episode_path,
                                                    video_name="episode=%04d" % episode,
                                                    fps=10)
            # Delete images
            shutil.rmtree(episode_path, ignore_errors=True)
            # Save models Q value NN function approximators
            for agent_id in env.agents.keys():
                ddqn_models[agent_id].save_policy(path=models_path + "/%s" % agent_id)

        print("- A:%d Episode %d - DONE in: %.3f min" % (n_agents, episode, (time.time() - start_t)/60))


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
    logs_path = "logs"


    params1 = {"n_agents": 1, "map_type": "small", "logs_path": logs_path, "n_episodes": EPISODES, "n_steps": STEPS,
               "batch_size": BATCH_SIZE, "lr": 0.0005, "gamma": 0.99, "epsilon": 0.15, "epsilon_decay": 0.999,
               "log": True}
    #
    # params2 = {"n_agents": 2, "map_type": "small", "logs_path": logs_path, "episodes": EPISODES, "n_steps": STEPS,
    #            "batch_size": BATCH_SIZE, "lr": 0.0015, "gamma": 0.99, "epsilon": 0.15, "epsilon_decay": 0.999,
    #            "log": False}
    #
    train_agents(**params1)
    # # processes = []
    # #
    # # for i, params in enumerate([params1.values(), params2.values()]):
    # #     p = Process(target=train_agents, args=params, name="Exp%d" % i)
    # #     p.start()
    # #     processes.append(p)
    # #     print(p.name)
    # #
    # # for p in processes:
    # #     p.join()
    # #
