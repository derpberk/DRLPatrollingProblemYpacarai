import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import interpolate
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from numpy import pi, exp, sqrt

class custom_filters():

    @staticmethod
    def EMA_filter(values, Alpha=0.6):
        filtered_values = []
        last_value = values[0]
        for value in values:
            actual_filtered_value = Alpha * value + (1 - Alpha) * last_value
            filtered_values.append(actual_filtered_value )
            last_value = actual_filtered_value

        return filtered_values

    @staticmethod
    def mean_filter(values, window=5):

        filtered_values = []
        for i in range(len(values)):
            ind = i + 1
            if ind < window:  # if we have not filled the window yet
                filtered_values.append(sum(values[:ind]) / ind)
            else:
                filtered_values.append(sum(values[(ind - window):ind]) / window)

        return filtered_values

    @staticmethod
    def median(dataset):
        data = sorted(dataset)
        index = len(data) // 2

        # If the dataset is odd
        if len(dataset) % 2 != 0:
            return data[index]

        # If the dataset is even
        return (data[index - 1] + data[index]) / 2

    def median_filter(self, values, window=5):

        filtered_values = []
        for i in range(len(values)):
            ind = i + 1
            if ind < window:  # if we have not filled the window yet
                filtered_values.append(self.median(values[:ind]))
            else:
                filtered_values.append(self.median(values[(ind - window):ind]))

        return filtered_values


def plot_trajectory(ax, x, y, z=None, colormap = 'jet', num_of_points = None, linewidth = 1, k = 3, plot_waypoints=False, markersize = 0.5):

    if z is None:
        #tck,u = interpolate.splprep([x,y],s=0.0, k=k)
        #x_i,y_i= interpolate.splev(np.linspace(0,1,num_of_points),tck)
        x_i,y_i= [x,y]
        points = np.array([x_i,y_i]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
        lc = LineCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
        lc.set_array(np.linspace(0,1,len(x_i)))
        fig2 = ax.add_collection(lc)
        plt.colorbar(fig2, ax=ax)
        if plot_waypoints:
            ax.plot(x,y,'.', color = 'black', markersize = markersize)
    else:
        tck,u = interpolate.splprep([x,y,z],s=0.0)
        x_i,y_i,z_i= interpolate.splev(np.linspace(0,1,num_of_points), tck)
        points = np.array([x_i,y_i,z_i]).T.reshape(-1,1,3)
        segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
        lc = Line3DCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
        lc.set_array(np.linspace(0,1,len(x_i)))
        ax.add_collection(lc)
        ax.scatter(x,y,z,'k')
        if plot_waypoints:
            ax.plot(x,y,'kx')

    ax.plot()

def gaussian_2d(s, k):
    """ generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s """
    probs = [exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)]
    kernel = np.outer(probs, probs)
    return kernel * (1 / np.max(kernel))  # *0.8

def put_kernel_in_map(map_shape, position, s, k):
    mask = np.zeros(map_shape)
    row, col = np.indices((2 * k + 1, 2 * k + 1))
    kernel = gaussian_2d(s, k)
    mask[row + position[0]-k, col + position[1]-k] = kernel
    return mask

def random_agent(env, num_of_episodes, directional = False):
    """ Reset the environment """
    env.reset()
    total_rew = 0
    rewards_by_episode = []

    # Reset metrics #
    """ 
    episodic_reward_vector = np.zeros(num_of_episodes)
    sum_of_interest = np.zeros(num_of_episodes)
    min_idleness_mean = np.zeros(num_of_episodes)
    idleness_mean = np.zeros(num_of_episodes)
    percentage_visited = np.zeros(num_of_episodes)
    exploration_reward_vector = np.zeros(num_of_episodes)
    step_count_vector = np.zeros(num_of_episodes)
    """

    episodic_reward_vector = []
    sum_of_interest = []
    min_idleness_mean = []
    idleness_mean = []
    percentage_visited = []
    exploration_reward_vector = []
    step_count_vector = []

    score_exploration = 0
    for episode in range(num_of_episodes):
        print(episode)
        state = env.reset()
        env.render()
        plt.pause(0.1)
        position = np.unravel_index(np.argmax(state[0], axis=None), state[0].shape)
        last_position = position
        action = env.action_space.sample()
        #print(episode)
        while not env.done:

            # Choose a valid action #
            if not directional:
                action = env.action_space.sample()
            val, attempted_position = env.check_action(action)
            valid = val and not np.array_equiv(attempted_position, last_position) if directional else val
            while not valid:
                action = env.action_space.sample()
                val, attempted_position = env.check_action(action)
                valid = val and not np.array_equiv(attempted_position, last_position) if directional else val

            #print(action)
            last_position = position
            # Random actions until done #
            next_state, reward, done, _ = env.step(action)

            total_rew = total_rew + reward
            score_exploration += env.exploration_reward
            if directional:
                position = np.unravel_index(np.argmax(next_state[0], axis=None), next_state[0].shape)
            #env.render()
            #plt.pause(0.1)

            sum_of_interest.append(env.sum_of_interest)
            min_idleness_mean.append(env.min_idleness_mean)
            idleness_mean.append(env.idleness_mean)
            percentage_visited.append(env.percentage_visited)
            episodic_reward_vector.append(total_rew)
            step_count_vector.append( env.step_count)
            exploration_reward_vector.append(score_exploration)
        """
        sum_of_interest[episode-1] = env.sum_of_interest
        min_idleness_mean[episode-1] = env.min_idleness_mean
        idleness_mean[episode-1] = env.idleness_mean
        percentage_visited[episode-1] = env.percentage_visited
        episodic_reward_vector[episode-1] = total_rew
        step_count_vector[episode-1] = env.step_count
        """
        total_rew = 0
        score_exploration = 0
        """if env.id == "Miopic Discrete Ypacarai":
            exploration_reward_vector[episode-1] = env.exploration_reward"""

        if env.step_count % 100 == 0:
            fig1, ax1 = plt.subplots(1, 1)
            #ax1.plot(agent.env.trajectory[:, 0], agent.env.trajectory[:, 1], '.', color='black', markersize=0.5)
            ax1.imshow(next_state[2], cmap='jet', vmin=0, vmax=1)
            plot_trajectory(ax1, env.trajectory[:, 1], env.trajectory[:, 0],colormap = 'binary', num_of_points=100, plot_waypoints=True)
            #plt.plot(episodes, mean_reward_vector, color='b')
            plt.title('Trayectoria')
            fig1.savefig('traj_directional'+str(env.step_count)+'.pdf',format = 'pdf')

            plt.close()

    return episodic_reward_vector, sum_of_interest, min_idleness_mean, idleness_mean, percentage_visited, exploration_reward_vector, step_count_vector  # NUEVO

def random_agent_lawnmower(env, num_of_episodes):
    """ Reset the environment """
    env.initial_position = [24, 2]
    env.reset()
    done = False

    # Reset metrics #
    """
        episodic_reward_vector = np.zeros(num_of_episodes)
        sum_of_interest = np.zeros(num_of_episodes)
        min_idleness_mean = np.zeros(num_of_episodes)
        idleness_mean = np.zeros(num_of_episodes)
        percentage_visited = np.zeros(num_of_episodes)
        exploration_reward_vector = np.zeros(num_of_episodes)
        step_count_vector = np.zeros(num_of_episodes)
    """
    episodic_reward_vector = []
    sum_of_interest = []
    min_idleness_mean = []
    idleness_mean = []
    percentage_visited = []
    exploration_reward_vector = []
    step_count_vector = []



    up = 0
    right = 1
    down = 2
    left = 3
    right_up = 4
    right_down = 5
    left_down = 6
    actions = [right_up, right, right_down, down, left_down]
    env.initial_position = [24,2]
    aux = 0
    action = up

    for episode in range(num_of_episodes):
        env.initial_position = [24,2]
        env.reset()
        done = False
        aux = 0
        action = up
        score = 0
        score_exploration = 0
        print(episode)
        #while not all(env.position == [54, 32]):
        while not done:
            val, attempted_position = env.check_action(action)
            if not val:
                for a in actions:
                    aux += 1
                    val, attempted_position = env.check_action(a)
                    if val:
                        if a == left_down:
                            next_state, reward, done, _ = env.step(a)
                            score += reward
                            score_exploration += env.exploration_reward
                        else:
                            next_state, reward, done, _ = env.step(a)
                            score += reward
                            score_exploration += env.exploration_reward
                            if action == up:
                                action = down
                                val, attempted_position = env.check_action(action)
                            elif action == down:
                                action = up
                                val, attempted_position = env.check_action(action)
                            if val:
                                next_state, reward, done, _ = env.step(action)
                                score += reward
                                score_exploration += env.exploration_reward
                        break
            else:
                next_state, reward, done, _ = env.step(action)
                score += reward
                score_exploration += env.exploration_reward


            #env.render()
            #plt.pause(0.1)
            episodic_reward = score
            sum_of_interest.append(env.sum_of_interest)
            min_idleness_mean.append(env.min_idleness_mean)
            idleness_mean.append(env.idleness_mean)
            percentage_visited.append(env.percentage_visited)
            episodic_reward_vector.append(episodic_reward)
            step_count_vector.append( env.step_count)
            exploration_reward_vector.append(score_exploration)
        """
        # Compute average metrics #
        episodic_reward = score
        sum_of_interest[episode-1] = env.sum_of_interest
        min_idleness_mean[episode-1] = env.min_idleness_mean
        idleness_mean[episode-1] = env.idleness_mean
        percentage_visited[episode-1] = env.percentage_visited
        episodic_reward_vector[episode-1] = episodic_reward
        step_count_vector[episode-1] = env.step_count
        if env.id == "Miopic Discrete Ypacarai":
            exploration_reward_vector[episode-1] = env.exploration_reward
        """
        """
        if env.step_count % 250 == 0:
            fig1, ax1 = plt.subplots(1, 1)
            #ax1.plot(agent.env.trajectory[:, 0], agent.env.trajectory[:, 1], '.', color='black', markersize=0.5)
            ax1.imshow(next_state[2], cmap='jet', vmin=0, vmax=1)
            plot_trajectory(ax1, env.trajectory[:, 1], env.trajectory[:, 0],colormap = 'binary', num_of_points=100, plot_waypoints=True)
            #plt.plot(episodes, mean_reward_vector, color='b')
            plt.title('Trayectoria')
            fig1.savefig('traj_lawnmower'+str(env.step_count)+'.pdf',format = 'pdf')

            plt.close()
        """
    return episodic_reward_vector, sum_of_interest, min_idleness_mean, idleness_mean, percentage_visited, exploration_reward_vector, step_count_vector

def evaluate_for_one_episode(agent, path_to_file, main_path = None, save_trajectory = None):
    """ Evaluate the agent. """

    # Agent in evaluation mode #
    agent.is_eval = True
    # Reset metrics #
    episodic_reward_vector = []
    exploration_reward_vector = []
    sum_of_interest = []
    min_idleness_mean = []
    idleness_mean = []
    percentage_visited = []

    agent.dqn.load_state_dict(torch.load(path_to_file, map_location=agent.device))

    done = False
    state = agent.env.reset()
    score = 0
    score_exploration = 0
    # Run an episode #
    fig1, ax1 = plt.subplots(1, 2)
    im = ax1[0].imshow(agent.env.information_map, cmap='jet', vmin=0, vmax=1)
    fig1.savefig(main_path +'\\inf_map.pdf',format = 'pdf')
    #plt.show()
    while not done:
        # Select the action
        action = agent.dqn(torch.FloatTensor(state).unsqueeze(0).to(agent.device)).argmax()
        action = action.detach().cpu().numpy()

        state, reward, done = agent.step(action)
        score += reward
        #score_exploration += agent.env.exploration_reward
        score_exploration = 0
        #agent.env.render()
        # Compute average metrics #
        episodic_reward = score
        episodic_reward_vector.append(episodic_reward)
        sum_of_interest.append(agent.env.sum_of_interest)
        min_idleness_mean.append(agent.env.min_idleness_mean)
        idleness_mean.append(agent.env.idleness_mean)
        percentage_visited.append(agent.env.percentage_visited)
        exploration_reward_vector.append(score_exploration)

    if save_trajectory is not None:# and (agent.env.step_count % save_trajectory == 0 or done):
        #ax1.plot(agent.env.trajectory[:, 0], agent.env.trajectory[:, 1], '.', color='black', markersize=0.5)
        #fig1, ax1 = plt.subplots(1, 1)
        im = ax1[1].imshow(state[2], cmap='jet', vmin=0, vmax=1)
        fig1.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)
        cb_ax = fig1.add_axes([0.83, 0.1, 0.02, 0.8])
        plt.colorbar(im, cax=cb_ax, ax=ax1)
        plot_trajectory(ax1[1], agent.env.trajectory[:, 1], agent.env.trajectory[:, 0],colormap = 'binary', num_of_points=100, plot_waypoints=True)
        #plt.plot(episodes, mean_reward_vector, color='b')
        #plt.colorbar()

        plt.show()
        plt.title('Trayectoria')
        if main_path is not None:
            fig1.savefig(main_path +'\\traj'+agent.env.id+str(agent.env.step_count)+'.pdf',format = 'pdf')

        else:
            fig1.savefig('traj'+str(agent.env.step_count)+'.pdf',format = 'pdf')

        plt.close()

    return episodic_reward_vector, sum_of_interest, min_idleness_mean, idleness_mean, percentage_visited, exploration_reward_vector  # NUEVO

"""
# Choose a valid action #
action = env.action_space.sample()
while not env.check_action(action):
    action = env.action_space.sample()

while not done:

    # Random actions until done #
    next_state, reward, done, _ = env.step(action)

    action = env.action_space.sample()
    if not env.check_action(action):
        action = env.action_space.sample()
        while not env.check_action(action):
            action = env.action_space.sample()

    env.render()
    plt.pause(0.1)
"""