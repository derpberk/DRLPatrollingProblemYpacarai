import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
class custom_filters():

    @staticmethod
    def EMA_filter(values, Alpha=0.6):
        filtered_values = []
        last_value = values[0]
        for value in values:
            filtered_values.append(Alpha * value + (1 - Alpha) * last_value)
            last_value = value

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
        ax.add_collection(lc)
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


def random_agent(env, num_of_episodes, directional = False):
    """ Reset the environment """
    env.reset()
    total_rew = 0
    rewards_by_episode = []

    for episode in range(num_of_episodes):
        print(episode)
        state = env.reset()
        position = np.unravel_index(np.argmax(state[0], axis=None), state[0].shape)
        last_position = position
        action = env.action_space.sample()
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

            total_rew = total_rew + 1
            if directional:
                position = np.unravel_index(np.argmax(next_state[0], axis=None), next_state[0].shape)

            env.render()
            plt.pause(0.1)

        rewards_by_episode.append(total_rew)
        total_rew = 0

    return rewards_by_episode

def random_agent_lawnmower(env):
    """ Reset the environment """
    env.initial_position = [24, 2]
    env.reset()
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
    while True:
        val, attempted_position = env.check_action(action)
        if not val:
            for a in actions:
                aux += 1
                val, attempted_position = env.check_action(a)
                if val:
                    if a == left_down:
                        next_state, reward, done, _ = env.step(a)
                    else:
                        next_state, reward, done, _ = env.step(a)
                        if action == up:
                            action = down
                            val, attempted_position = env.check_action(action)
                        elif action == down:
                            action = up
                            val, attempted_position = env.check_action(action)
                        if val:
                            next_state, reward, done, _ = env.step(action)
                    break
        else:
            next_state, reward, done, _ = env.step(action)

        env.render()
        plt.pause(0.1)

    return
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