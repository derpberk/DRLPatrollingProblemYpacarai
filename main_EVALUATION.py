from Environment.Miopic_IPP_Ypacarai import DiscreteIPP as Miopic_IPP
from Environment.Soft_IPP_Ypacarai import DiscreteIPP as Soft_IPP
from DeepAgent.Agent.DuelingDQNAgent import DuelingDQNAgent
import numpy as np
import matplotlib.pyplot as plt


# Load the navigation map #
navigation_map = np.genfromtxt('Environment/example_map.csv', delimiter=',')

# HYPERPARAMETERS PARAMETERS #
batch_size = 64
experience_replay_size = 100000
target_update = 1000
num_of_episodes = 20000
num_of_episodes_for_evaluation = 25

### Para cada entrenamiento cambiar la carpeta runs/MiopicYpacaraiExperiment o en su caso la variable env###
episodes = np.arange(2000, num_of_episodes + 1, 2000)
paths_to_file = ['runs/MiopicYpacaraiExperiment/Policy_Episode_' + str(i)+'.pth' for i in episodes]
paths_to_file.append('runs/MiopicYpacaraiExperiment/BestPolicy.pth')
mean_reward_vector = np.zeros(len(paths_to_file))
std_reward_vector = np.zeros(len(paths_to_file))
for i, path_to_file in enumerate(paths_to_file):
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
                     discovery_reward=0
                     )
   
    # Reset the environment
    env.reset()
    done = False

    Agent = DuelingDQNAgent(env=env,
                            memory_size=experience_replay_size,
                            batch_size=batch_size,
                            target_update=target_update,
                            epsilon_values=(1.0, 0.05),
                            epsilon_interval=(0, 0.33),
                            gamma=0.99,
                            lr=1e-4,
                            max_pool=True,
                            train_every=15,
                            save_every=num_of_episodes // 10)

    reward_vector = Agent.evaluate(num_of_episodes_for_evaluation, path_to_file)
    mean_reward_vector[i] = np.mean(reward_vector)
    std_reward_vector[i] = np.std(reward_vector)
"""
mean_reward_vector = np.asarray([-20.603, -18.55477503, -17.69250267,-15.29503493, -10.55477503])
std_reward_vector = np.asarray([0.603, 1.55477503, 2.69250267,0.29503493, 0.55477503])
episodes = np.arange(2000, 10000 + 1, 2000)
"""
fig = plt.figure()
plt.plot(episodes, mean_reward_vector, color='b')
plt.title('Evolution of the training')
plt.ylabel('Means of the reward')
plt.xlabel('Episodes')
plt.fill_between(episodes, mean_reward_vector + std_reward_vector,
                 mean_reward_vector - std_reward_vector, color='lightblue')
fig.savefig('means_and_std.pdf',format = 'pdf')