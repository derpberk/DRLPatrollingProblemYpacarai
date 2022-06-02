from Environment.Miopic_IPP_Ypacarai import DiscreteIPP as Miopic_IPP
from DeepAgent.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
import matplotlib.pyplot as plt

# Load the navigation map #
navigation_map = np.genfromtxt('Environment/example_map.csv', delimiter=',')

# Create the environment #
env = Miopic_IPP(scenario_map=navigation_map,
                 initial_position=np.array([26, 21]),
                 battery_budget=100,
                 detection_length=2,
                 random_information=True,
                 seed=1,
                 num_of_kilometers=100,
                 recovery=0.03,
                 attrition=0.05,
                 collisions_allowed=True,
                 num_of_allowed_collisions=20,
                 )

# Reset the environment
env.reset()
done = False

# Create the environment #

# HYPERPARAMETERS PARAMETERS #
batch_size = 64
experience_replay_size = 100000
target_update = 1000
num_of_episodes = 40000

Agent = DuelingDQNAgent(env=env,
                        memory_size=experience_replay_size,
                        batch_size=batch_size,
                        target_update=target_update,
                        epsilon_values=(1.0, 0.05),
                        epsilon_interval=(0, 0.33),
                        gamma=0.99,
                        lr=1e-4,
                        train_every=15,
                        save_every=num_of_episodes // 10)

losses, episodic_reward_vector = Agent.train(num_of_episodes)

"""
# Choose a valid action #
action = env.action_space.sample()
while not env.check_action(action):
    action = env.action_space.sample()

while not done:

    # Random actions until done #
    next_state, reward, done, _ = env.step(action)

    if not env.check_action(action):
        action = env.action_space.sample()
        while not env.check_action(action):
            action = env.action_space.sample()

    env.render()
    plt.pause(0.1)
"""
