import gym
import numpy as np
import matplotlib.pyplot as plt
from Environment.groundtruthgenerator import GroundTruth
from Environment.utils import random_agent, put_kernel_in_map


class DiscreteIPP(gym.Env):
    environment_name = "Discrete Informative Path Planning"

    def __init__(self, scenario_map, initial_position=None, battery_budget=100,
                 detection_length=2, random_information=True, seed=0, num_of_kilometers=30,
                 collisions_allowed=False, num_of_allowed_collisions=10, attrition=0, recovery=0.01):

        self.id = "Discrete Ypacarai"

        # Map of the environment #
        self.scenario_map = scenario_map
        self.map_size = self.scenario_map.shape
        self.map_lims = np.array(self.map_size) - 1
        self.posibles = np.asarray(np.nonzero(self.scenario_map)).T

        """ Generate ground truth based on the Shekel function """
        self.gt = GroundTruth(grid=1 - scenario_map, resolution=1, max_number_of_peaks=4, is_bounded=True, seed=seed)
        self.gt.reset()
        self.fixed_gt = self.gt.read()
        self.random_information = random_information
        self.information_map = None
        self.information_importance = None
        self.visited_map = None
        # metrics of interest
        self.sum_of_interest = 0
        self.min_idleness_mean = np.inf
        self.idleness_mean = None  # the average idleness value of each box
        self.percentage_visited = None  # percentage of the map visited

        """ Action spaces for gym convenience """
        self.action_space = gym.spaces.Discrete(8)
        self.action_size = 8

        """ Observation_space """
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, self.map_size[0], self.map_size[1]),
                                                dtype=np.float32) # IMPORTANTE: shape antes era (4, self....

        self.detection_length = detection_length

        self.state = None
        self.next_state = None
        self.reward = None
        self.done = False

        self.initial_position = initial_position
        self.step_count = 0

        self.collision_penalization = 1

        self.seed = seed
        np.random.seed(self.seed)

        self.fig = None
        self.axs = None
        self.fig_trajectory = None
        self.im0_trajectory = None
        self.im_traj = None

        # Initialization of the random map information information #

        self.place_information()

        self.position = None
        self.trajectory = None
        self.place_agent()

        # Battery budget
        self.battery = battery_budget

        self.max_num_of_movements = int(2 * (num_of_kilometers / 30) * self.map_size[0])  # About 30 km
        self.battery_cost = 100 / self.max_num_of_movements
        self.recovery_rate = recovery
        self.interest_permanent_loss_rate = attrition

        # Number of allowed collisions before ending an episode
        self.num_of_allowed_collisions = num_of_allowed_collisions
        self.num_of_collisions = 0
        self.collisions_allowed = collisions_allowed

        self.reset()

    def reset(self):

        self.place_agent()

        self.place_information()

        self.num_of_collisions = 0
        self.battery = 100
        self.reward = None
        self.done = False
        self.next_state = None
        self.step_count = 0

        self.visited_map = None
        self.sum_of_interest = 0
        self.min_idleness_mean = np.inf
        self.idleness_mean = None  # the average idleness value of each box
        self.percentage_visited = None  # percentage of the map visited

        self.state, self.reward = self.process_state_and_reward()

        return self.state

    def place_agent(self):
        """ Place the agent in a random place. """
        if self.initial_position is None:
            indx = np.random.randint(0, len(self.posibles))
            self.position = self.posibles[indx]
            self.trajectory = np.atleast_2d(self.position)
        else:
            self.position = np.copy(self.initial_position)
            self.trajectory = np.atleast_2d(self.position)

    def place_information(self):
        """ Place the information """

        if self.random_information:
            self.gt.reset()
            self.information_map = self.gt.read()
        else:
            self.information_map = np.copy(self.fixed_gt)
        """
        ################### TESTS: Add a Gaussian distribution peak at the position pos.
        pos = [45, 25]
        s, k = 2.5, 11  # generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
        mask = put_kernel_in_map(self.scenario_map.shape, pos, s, k)
        self.information_map = np.clip(self.information_map + mask, 0, 1) * self.scenario_map
        
        self.fixed_inf_map=self.information_map
        ###################################
        """
        self.information_importance = np.copy(self.scenario_map)

    def render(self, img_type='None'):

        plt.ion()

        if self.fig is None:

            self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 3))
            self.im0 = self.axs[0].imshow(self.state[0], cmap='gray', vmin=0, vmax=1)
            self.axs[0].set_title('Position')
            self.im1 = self.axs[1].imshow(self.state[1], cmap='gray', vmin=0, vmax=1)
            self.axs[1].set_title('Navigation map')
            self.im2 = self.axs[2].imshow(self.state[2], cmap='jet', vmin=0, vmax=1)
            self.axs[2].set_title('Importance Map')

        else:

            self.im0.set_data(self.state[0])
            self.im1.set_data(self.state[1])
            self.im2.set_data(self.state[2])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def step(self, desired_action):

        self.step_count += 1

        # Check the feasibility of the action #
        val, new_position = self.check_action(desired_action)

        if val:  # Valid action
            # IF valid, update the position
            self.position = new_position
            self.trajectory = np.row_stack((self.trajectory, self.position))
        elif self.collisions_allowed:
            self.num_of_collisions += 1
            # print(desired_action, self.num_of_collisions, self.step_count)

        # Process state and reward #

        self.state, self.reward = self.process_state_and_reward(valid=val)

        # Compute the battery consumption #
        self.battery -= self.battery_cost if desired_action < 4 else 1.4142 * self.battery_cost

        if self.collisions_allowed:
            done_due_to_collisions = self.num_of_collisions >= self.num_of_allowed_collisions
        else:
            done_due_to_collisions = not val  # if we are not allowed to collision

        # Check the episodic end condition
        self.done = self.battery <= self.battery_cost or done_due_to_collisions

        return self.state, self.reward, self.done, {}

    def check_action(self, desired_action):

        v = self.action2vector(desired_action)
        valid = False

        attempted_position = self.position + v

        if self.scenario_map[attempted_position[0], attempted_position[1]] == 1:
            valid = True
        else:
            valid = False

        return valid, attempted_position

    def reward_function(self, collision_free, coverage_area):

        if collision_free:
            sum_of_information = np.sum(coverage_area * self.information_map * self.information_importance)
            reward = sum_of_information / (self.detection_length ** 2 * np.pi)
        else:
            reward = -self.collision_penalization

        return reward

    def process_state_and_reward(self, valid=True):

        state = np.zeros(shape=(3, self.scenario_map.shape[0], self.scenario_map.shape[1]))

        # State - position #

        state[0, self.position[0], self.position[1]] = 1.0

        # State - boundaries #
        state[1] = np.copy(self.scenario_map)

        # State - coverage area #
        x = np.arange(0, self.scenario_map.shape[1])
        y = np.arange(0, self.scenario_map.shape[0])

        # Compute the circular mask (area) of the state 3 #
        cx = self.position[1]
        cy = self.position[0]
        r = self.detection_length

        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
        mask = mask.astype(float)

        if self.visited_map is None:
            self.visited_map = np.copy(mask)
        else:
            self.visited_map = np.clip(self.visited_map + mask, 0, 1)

        # Reward function #

        reward = self.reward_function(collision_free=valid, coverage_area=mask)
        if reward != -self.collision_penalization:
            self.sum_of_interest = self.sum_of_interest + np.sum(mask * self.information_map * self.information_importance)
        # Update the information map

        # Redraw the importance in the covered area #
        self.information_importance = np.clip(self.information_importance - mask * self.information_importance, 0, 1)
        # The information map is decreased with the attrition factor
        self.information_map = np.clip(self.information_map - mask * self.interest_permanent_loss_rate * self.gt.read(), 0, 1)
        #self.information_map = np.clip(self.information_map - mask * self.interest_permanent_loss_rate * self.fixed_inf_map, 0, 1) #for when there is a gaussian peak

        # State - Relative Importance map #
        state[2] = self.information_map * self.information_importance

        # Recover the idleness of the information #
        self.information_importance = np.clip(self.information_importance + self.recovery_rate, 0, 1)

        self.idleness_mean = np.sum(state[2])/np.count_nonzero(self.scenario_map)
        if self.idleness_mean < self.min_idleness_mean:
            self.min_idleness_mean = self.idleness_mean

        self.percentage_visited = np.count_nonzero(self.visited_map)/np.count_nonzero(self.scenario_map)
        # print('Soft_Ypacarai', self.percentage_visited, self.idleness_mean)

        return state, reward

    @staticmethod
    def action2vector(action):

        if action == 0:
            vector = np.asarray([-1, 0])
        elif action == 1:
            vector = np.asarray([0, 1])
        elif action == 2:
            vector = np.asarray([1, 0])
        elif action == 3:
            vector = np.asarray([0, -1])
        elif action == 4:
            vector = np.asarray([-1, 1])
        elif action == 5:
            vector = np.asarray([1, 1])
        elif action == 6:
            vector = np.asarray([1, -1])
        elif action == 7:
            vector = np.asarray([-1, -1])

        return vector

    def evaluate_trajectory(self, trajectory):

        """ Reset the environment """
        self.reset()
        R = 0
        for i in range(len(trajectory)):

            _, r, d, _ = self.step(trajectory[i])
            R += r

            if d:
                break

        return R
    """
    def random_agent(self, num_of_episodes, allowed_collisions=None):

        ### Reset the environment ###
        self.reset()
        total_rew = 0
        rewards_by_episode = []
        fig1, ax1 = plt.subplots(1,1)

        if allowed_collisions is not None:  # we can change the number of allowed collisions also in this method
            self.num_of_allowed_collisions = allowed_collisions

        for episode in range(num_of_episodes):
            self.reset()
            while not self.done:
                a_ = np.random.randint(0, 7)

                s_, r_, d_, _ = env.step(a_)
                total_rew += r_
                self.render()
                self.show_trajectory()
                plt.pause(0.1)

            rewards_by_episode.append(total_rew)
            total_rew = 0

            plot_trajectory(ax1, self.trajectory[:, 0], self.trajectory[:, 1], num_of_points = 100, plot_waypoints=True)

        return rewards_by_episode
"""
    def show_trajectory(self):

        plt.ion()

        if self.fig_trajectory is None:
            self.fig_trajectory = plt.figure()
            self.im_traj = np.copy(self.scenario_map)
            self.im_traj[self.trajectory[-1, 0], self.trajectory[-1, 1]] = 0.6
            self.im0_trajectory = plt.imshow(self.im_traj, cmap='jet', vmin=0, vmax=1)
            self.im_traj[self.trajectory[-1, 0], self.trajectory[-1, 1]] = 0.3
            plt.title('Trajectory')
        else:
            # self.im_traj[self.position[0], self.position[1]] = 0.6
            self.im_traj[self.trajectory[-1, 0], self.trajectory[-1, 1]] = 0.6
            self.im0_trajectory.set_data(self.im_traj)
            self.im_traj[self.trajectory[-1, 0], self.trajectory[-1, 1]] = 0.3

        im_traj_copy = self.im_traj

        if self.done:
            self.im_traj = np.copy(self.scenario_map)

        self.fig_trajectory.canvas.draw()
        self.fig_trajectory.canvas.flush_events()

        return im_traj_copy


if __name__ == "__main__":
    """ Ejemplo de uso del escenario"""

    """ Se carga el escenario."""
    """ scenario_map = mapa de obstaculos.
        detection_length = radio de la circunferencia de detección en píxeles
        initial_position = posicion inicial de despliegue
        seed = semilla del mapa. Determina el mapa inicial.
        random_information = determina si al hacer reset cambiamos el mapa. Si es False, al hacer reset volvemos al mismo
        mapa inicial determinado por la seqmilla.
        num_of_kilometers = determina el numero de kilómetros máximos que hace el dron. 
        Se traduce en movimientos. 30km ~> 100 movs)
         
        """
    plt.switch_backend('TkAgg')
    my_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
    env = DiscreteIPP(scenario_map=my_map,
                      detection_length=2,
                      initial_position=np.array([26, 21]),
                      seed=1,
                      random_information=True,
                      num_of_kilometers=120,
                      collisions_allowed=True,
                      num_of_allowed_collisions=8,
                      recovery=0.03,
                      attrition=0.05)

    s = env.reset()
    Random_agent_mean_rewards = random_agent(env, 100000, directional=True)
    np.savetxt('Random_agent_mean_rewards.csv', Random_agent_mean_rewards, delimiter= ' ')
    plt.figure()
    plt.plot(env.trajectory[:, 0], env.trajectory[:, 1])
    plt.show(block=True)
"""
    c_filters = custom_filters()
    Random_agent_mean_rewards = np.genfromtxt('Random_agent_mean_rewards.csv')
    Random_agent_mean_rewards2 = c_filters.EMA_filter(Random_agent_mean_rewards, 0.4)
    Random_agent_mean_rewards3 = c_filters.median_filter(Random_agent_mean_rewards, 6)
    Random_agent_mean_rewards4 = c_filters.mean_filter(Random_agent_mean_rewards, 6)

    plt.figure(1)
    plt.plot(Random_agent_mean_rewards)
    plt.plot(Random_agent_mean_rewards2)
    plt.title("Rewards filtered with the EMA filter")
    plt.legend(['Without filter', 'Filtered'])

    plt.figure(2)
    plt.plot(Random_agent_mean_rewards)
    plt.plot(Random_agent_mean_rewards3)
    plt.title("Rewards filtered with the median filter")
    plt.legend(['Without filter', 'Filtered'])

    plt.figure(3)
    plt.plot(Random_agent_mean_rewards)
    plt.plot(Random_agent_mean_rewards4)
    plt.title("Rewards filtered with the mean filter")
    plt.legend(['Without filter', 'Filtered'])

    plt.show(block=True)
    # plt.interactive(False)
    """
"""
    total_r = 0
    t = 0
    R_vec = [total_r]

    while not env.done:


        a = np.random.randint(0, 8)


        s, r, d, _ = env.step(a)

        env.render()
        plt.pause(0.1)

        total_r += r

        R_vec.append(total_r)

        t += 1


    print('Recompensa', total_r, 'Timesteps ', t)

"""
