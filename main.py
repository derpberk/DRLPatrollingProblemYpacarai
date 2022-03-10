from Environment.Soft_IPP_Ypacarai import DiscreteIPP
import numpy as np
import matplotlib.pyplot as plt

# Load the navigation map #
navigation_map = np.genfromtxt('Environment/example_map.csv',delimiter=',')

# Create the environment #
env = DiscreteIPP(scenario_map = navigation_map, initial_position=np.array([26,21]), battery_budget=100,
                 detection_length = 2, random_information=True, seed = 1, num_of_kilometers = 100)

# Reset the environment
env.reset()
done = False

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