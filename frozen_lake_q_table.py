"""Create a solution to the frozen lake game using a Q-table

    Parameters:
        max_steps
        num_episodes
        min_exploration_rate
        max_exploration_rate
        exploration_decay_rate
        learning_rate
        discount_rate

    Approach:
    Initialize Q-table to be naive
    For each episode:
        For each step:
            with some probability of exploring:
                pick a random action
            else:
                pick the action with the maximum Q-value
                update Q-value for the state and action:
                    Q(s, a) := Q(s, a) + alpha*(r + gamma * max(Q(s', :) - Q(s, a))
        Update exploration probability

"""

import gym
import numpy as np
from gym.envs.registration import register

register(
    id="FrozenLake-v1",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"map_name": "8x8", "is_slippery": False},
    max_episode_steps=100,
    reward_threshold=0.8196,  # optimum = .8196, changing this seems have no influence
)


class QTableAgent:
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.max_steps = 100
        self.num_episodes = 15000
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0
        self.exploration_decay_rate = 0.999
        self.learning_rate = 0.8
        self.discount_rate = 0.95

    def exploration_rate(self, episode):
        return (
            self.min_exploration_rate
            + (self.max_exploration_rate - self.min_exploration_rate)
            * self.exploration_decay_rate ** episode
        )

    def update_q_table(self, state, new_state, action, reward):
        self.q_table[state, action] = self.q_table[
            state, action
        ] + self.learning_rate * (
            reward
            + self.discount_rate * np.max(self.q_table[new_state])
            - self.q_table[state, action]
        )

    def learn(self):
        rewards = []
        self.exploration_decay_rate = 0.999
        for episode in range(self.num_episodes):
            state = self.env.reset()
            cumulative_rewards = 0
            for step in range(self.max_steps):
                if np.random.random() > self.exploration_rate(episode):
                    action = np.argmax(self.q_table[state])
                else:
                    action = self.env.action_space.sample()
                new_state, reward, done, _ = self.env.step(action)
                if done and reward == 0:
                    reward = -1
                self.update_q_table(state, new_state, action, reward)
                state = new_state
                cumulative_rewards += reward
                if done:
                    break

            rewards.append(cumulative_rewards)
            print(
                f"Episode {episode} reward: {cumulative_rewards}, at step: {step},"
                f" with win rate of {sum(rewards[-1000:]) / len(rewards[-1000:])}"
            )

    def play(self):
        state = self.env.reset()
        cumulative_rewards = 0
        env.render()
        for _ in range(self.max_steps):
            action = np.argmax(self.q_table[state])
            state, reward, done, _ = self.env.step(action)
            cumulative_rewards += reward
            env.render()
            if done:
                print(f"Done with reward: {cumulative_rewards}")
                break


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    agent = QTableAgent(env)
    agent.learn()
    agent.play()
