from Environment.Soft_IPP_Ypacarai import DiscreteIPP as Soft_IPP
from Environment.Miopic_IPP_Ypacarai import DiscreteIPP as Miopic_IPP
from DeepAgent.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
import matplotlib.pyplot as plt

# Load the navigation map #
navigation_map = np.genfromtxt('Environment/example_map.csv', delimiter=',')

# Create the environment #
env = Soft_IPP(scenario_map=navigation_map, initial_position=np.array([26, 21]), battery_budget=100,
                  detection_length=2, random_information=True, seed=1, num_of_kilometers=100)

env_miopic = Miopic_IPP(scenario_map=navigation_map, initial_position=np.array([26, 21]), battery_budget=100,
                  detection_length=2, random_information=True, seed=1, num_of_kilometers=100)

# Reset the environment
env.reset()
done = False

# Create the environment #

batch_size = 32
experience_replay_size = 10000
target_update = 1000

Agent = DuelingDQNAgent(env, experience_replay_size, batch_size, target_update)

num_of_episodes = 15000
losses, episodic_reward_vector = Agent.train(num_of_episodes)
plt.figure(1)
plt.plot(losses)
plt.figure(2)
plt.plot(episodic_reward_vector)
plt.show(block=True)
print('done')
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