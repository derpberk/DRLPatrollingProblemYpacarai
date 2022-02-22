import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_trajectory(ax, x, y, z=None, colormap = 'jet', num_of_points = None, linewidth = 1, k = 3, plot_waypoints=False, markersize = 0.5):

    if z is None:
        tck,u = interpolate.splprep([x,y],s=0.0, k=k)
        x_i,y_i= interpolate.splev(np.linspace(0,1,num_of_points),tck)
        points = np.array([x_i,y_i]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
        lc = LineCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
        lc.set_array(np.linspace(0,1,len(x_i)))
        ax.add_collection(lc)
        if plot_waypoints:
            ax.plot(x,y,'.', color = 'black', markersize = markersize)
    else:
        tck,u =interpolate.splprep([x,y,z],s=0.0)
        x_i,y_i,z_i= interpolate.splev(np.linspace(0,1,num_of_points), tck)
        points = np.array([x_i,y_i,z_i]).T.reshape(-1,1,3)
        segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
        lc = Line3DCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
        lc.set_array(np.linspace(0,1,len(x_i)))
        ax.add_collection(lc)
        ax.scatter(x,y,z,'k')
        if plot_waypoints:
            ax.plot(x,y,'kx')