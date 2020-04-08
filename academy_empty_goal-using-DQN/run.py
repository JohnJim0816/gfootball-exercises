# -*- coding: utf-8 -*-
# @Author: Jiang Ji
# @Date:   2020-04-07 15:09:53
# @Last Modified by:   Jiang Ji
# @Last Modified time: 2020-04-08 16:37:45
# coding=utf-8
import gfootball.env as football_env
from deep_q_network import DQN


def plot_steps(steps):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    plt.plot(np.arange(len(steps)), steps)
    plt.ylabel('steps')
    plt.xlabel('episodes')
    plt.savefig("steps.png")
    plt.show()


def plot_rewards(rewards):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    plt.plot(np.arange(len(rewards)), rewards)
    plt.ylabel('rewards')
    plt.xlabel('episodes')
    plt.savefig("./rewards.png")
    plt.show()


if __name__ == "__main__":

    env = football_env.create_environment(env_name="academy_empty_goal_close",
                                          representation='simple115',
                                          number_of_left_players_agent_controls=1,
                                          stacked=False, logdir='/tmp/football',
                                          write_goal_dumps=False,
                                          write_full_episode_dumps=False, render=False)
    n_actions = 19 # number of actions to choose
    n_features = 115 # dims of observation
    RL = DQN(19, 115,
             learning_rate=0.01,
             reward_decay=0.9,
             e_greedy=0.98,
             replace_target_iter=200,
             memory_size=2000,
             # output_graph=True
             )

    max_episodes = 300

    steps = []
    rewards = []

    for i_episode in range(1, max_episodes+1):
        observation = env.reset()
        step = 0
        while True:
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            # if (step > 100) and (step % 5 == 0):
            #     RL.learn()
            RL.learn()
            observation = observation_
            step += 1
            # break while loop when end of this episode
            if done:
                break
        rewards.append(reward)
        steps.append(step)

    plot_steps(steps)
    plot_rewards(rewards)
    RL.plot_cost()
