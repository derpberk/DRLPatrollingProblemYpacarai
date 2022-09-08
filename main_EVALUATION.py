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
num_of_episodes_for_evaluation = 100
paths_to_file = []
"""
### Para cada entrenamiento cambiar la carpeta runs/MiopicYpacaraiExperiment o en su caso la variable env###
main_path = "D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\Archivoscsv_"
paths_to_file.append('D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\BestPolicy.pth')
paths_to_file.append('D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\FINALPolicy.pth')
mean_reward_vector = np.zeros(len(paths_to_file))
std_reward_vector = np.zeros(len(paths_to_file))
# metrics of interest
sum_of_interest = 0
min_idleness_mean = np.inf
idleness_mean = None  # the average idleness value of each box
percentage_visited = None  # percentage of the map visited
"""
paths_to_file = []
### Para cada entrenamiento cambiar la carpeta runs/MiopicYpacaraiExperiment o en su caso la variable env###
episodes = np.arange(2000, num_of_episodes + 1, 2000)
main_path = "D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\Archivoscsv_policies"
paths_to_file = ['D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\Policy_Episode_' + str(i)+'.pth' for i in episodes]
paths_to_file.append('D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\BestPolicy.pth')
mean_reward_vector = np.zeros(len(paths_to_file))
std_reward_vector = np.zeros(len(paths_to_file))
sum_of_interest_vector = np.zeros(len(paths_to_file))
min_idleness_mean_vector = np.zeros(len(paths_to_file))
idleness_mean_vector = np.zeros(len(paths_to_file))
percentage_visited_vector = np.zeros(len(paths_to_file))
for i, path_to_file in enumerate(paths_to_file):
    # Create the environment #

    env = Soft_IPP(scenario_map=navigation_map,
                   initial_position=np.array([26, 21]),
                   battery_budget=100,
                   detection_length=2,
                   random_information=True,
                   seed=2,
                   num_of_kilometers=100,
                   recovery=0.03,
                   attrition=0.05,
                   collisions_allowed=False,
                   num_of_allowed_collisions=20,
                   )

    # Reset the environment
    env.reset()
    done = False

    Agent = DuelingDQNAgent(env=env,
                            memory_size=experience_replay_size,
                            batch_size=batch_size,
                            target_update=target_update,
                            epsilon_values=(0.0, 0.0),
                            epsilon_interval=(0, 0.33),
                            gamma=0.99,
                            lr=1e-4,
                            max_pool=False,
                            train_every=15,
                            save_every=num_of_episodes // 10)
    #path_to_file = 'D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\BestPolicy.pth'
    #reward_vector, sum_of_interest, min_idleness_mean, idleness_mean, percentage_visited = Agent.evaluate(num_of_episodes_for_evaluation, path_to_file, save_trajectory=i)
    #np.savetxt(main_path + '\metrics_dqn'+str(i)+'.csv', np.column_stack([reward_vector, sum_of_interest, min_idleness_mean, idleness_mean, percentage_visited]))

    reward_vector,_,_a,_b,_c = Agent.evaluate(num_of_episodes_for_evaluation, path_to_file, save_trajectory=i)
    mean_reward_vector[i] = np.mean(reward_vector)
    std_reward_vector[i] = np.std(reward_vector)
    sum_of_interest_vector[i] = np.mean(_)
    min_idleness_mean_vector[i] = np.mean(_a)
    idleness_mean_vector[i] = np.mean(_b)
    percentage_visited_vector[i] = np.mean(_c)

np.savetxt(main_path + '\metrics_dqn100ep0.csv', np.column_stack([mean_reward_vector, std_reward_vector, sum_of_interest_vector, min_idleness_mean_vector, idleness_mean_vector, percentage_visited_vector]))

fig, ax = plt.figure()
plt.plot(episodes, mean_reward_vector, color='b')
plt.title('Evolution of the training')
plt.ylabel('Means of the reward')
plt.xlabel('Episodes')
plt.fill_between(episodes, mean_reward_vector + std_reward_vector,
                 mean_reward_vector - std_reward_vector, color='lightblue')
fig.savefig('means_and_std.pdf',format = 'pdf')

### Para cada entrenamiento cambiar la carpeta runs/MiopicYpacaraiExperiment o en su caso la variable env###
episodes = np.arange(2000, num_of_episodes + 1, 2000)
#paths_to_file = ['runs/MiopicYpacaraiExperiment/Policy_Episode_' + str(i)+'.pth' for i in episodes]
paths_to_file.append('D:\Desktop\ADS\GIERM\TFG\experimentos\MiopicYpacaraiExperiment\FINALPolicy.pth')
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
                     collisions_allowed=False,
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
                            max_pool=False,
                            train_every=15,
                            save_every=num_of_episodes // 10)
    path_to_file = 'D:\Desktop\ADS\GIERM\TFG\experimentos\SoftYpacaraiExperiment_LeakyRELU\Policy_Episode_2000.pth'
    reward_vector = Agent.evaluate(num_of_episodes_for_evaluation, path_to_file, save_trajectory=i)
    mean_reward_vector[i] = np.mean(reward_vector)
    std_reward_vector[i] = np.std(reward_vector)
"""
mean_reward_vector = np.asarray([-20.603, -18.55477503, -17.69250267,-15.29503493, -10.55477503])
std_reward_vector = np.asarray([0.603, 1.55477503, 2.69250267,0.29503493, 0.55477503])
episodes = np.arange(2000, 10000 + 1, 2000)
"""
fig, ax = plt.figure()
plt.plot(episodes, mean_reward_vector, color='b')
plt.title('Evolution of the training')
plt.ylabel('Means of the reward')
plt.xlabel('Episodes')
plt.fill_between(episodes, mean_reward_vector + std_reward_vector,
                 mean_reward_vector - std_reward_vector, color='lightblue')
fig.savefig('means_and_std.pdf',format = 'pdf')