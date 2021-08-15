import numpy as np
from numpy.random import normal as normal
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from utils.data import *
from utils.flags import FLAGS


def update(ifrm, xa, ya, za,scatters):
    scatters.set_data(xa[ifrm], ya[ifrm])
    scatters.set_3d_properties(za[ifrm])

def animate(mocap_seq,save=False):
    print("Number of frame", len(mocap_seq))
    print(mocap_seq)

    all_3d_coords = mocap_seq.reshape(-1, 3, 41)
    print(all_3d_coords.shape)

    nfr = len(mocap_seq) # Number of frames
    fps = 120 # Frame rate
    markers = all_3d_coords.shape[2]

    xs = []
    ys = []
    zs = []

    
    for frame in range(nfr):
        #print("Frame: ",frame, all_3d_coords[frame][0])

        xs.append(np.array(all_3d_coords[frame][0]))
        ys.append(np.array(all_3d_coords[frame][1]))
        zs.append(np.array(all_3d_coords[frame][2]))

    #print(xs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatters, = ax.plot([], [], [], "o", c= '#000000', markersize=2)

    ## Creating plot
    #scatters = ax.scatter3D(xs, ys, zs,
    #                    alpha = 0.8,
    #                    color = 'green',
    #                    marker ='o')
 


    ## Find which points are present

    #key_point_arm = []
    #for point in list([0, 1, 2, 7, 8, 9]):
    #    if all_3d_coords[step][0][point] != 0 and all_3d_coords[step][0][point + 1] != 0:
    #        if all_3d_coords[step][1][point] != 0 and all_3d_coords[step][1][point + 1] != 0:
    #            if all_3d_coords[step][2][point] != 0 and all_3d_coords[step][2][point + 1] != 0:
    #                key_point_arm.append(point)
    #key_point_arm = np.array(key_point_arm)

    #key_point_leg = []
    #for point in list([27, 34]):
    #    if all_3d_coords[step][0][point] != 0 and all_3d_coords[step][0][point + 1] != 0:
    #        if all_3d_coords[step][1][point] != 0 and all_3d_coords[step][1][point + 1] != 0:
    #            if all_3d_coords[step][2][point] != 0 and all_3d_coords[step][2][point + 1] != 0:
    #                key_point_leg.append(point)
    #key_point_leg = np.array(key_point_leg)

    ## Add lines in between

    #for point in key_point_arm:
    #    xline = all_3d_coords[step][0][point:point + 2]
    #    yline = np.add(all_3d_coords[step][1][point:point + 2], (step - start_frame) * coef)
    #    zline = all_3d_coords[step][2][point:point + 2]
    #    ax.plot(xline, yline, zline, c=colors[0])

    #for point in key_point_leg:
    #    xline = all_3d_coords[step][0][point:point + 3:2]
    #    yline = np.add(all_3d_coords[step][1][point:point + 3:2], (step - start_frame) * coef)
    #    zline = all_3d_coords[step][2][point:point + 3:2]
    #    ax.plot(xline, yline, zline, c=colors[3])


    ax.set_title('15 (out of 41) markers are missing in the boxing motion')
    ax.set_xlim(-750,750)
    ax.set_ylim(-750,750)
    ax.set_zlim(-750,750)
    ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs,ys,zs,scatters), interval=1000/fps)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=120, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('boxing-noisy.mp4', writer=writer)
    plt.show()

if __name__ == '__main__':
    
    #data = generate_data(100, 2)
    #main(data)
    #seq = read_test_seq_from_binary(FLAGS.data_dir + '/test_seq/boxing.binary')
    #print(seq)
    original_seq = np.genfromtxt(FLAGS.data_dir + '/test_seq/boxing.binary_original.csv', delimiter=',')
    noisy_seq = np.genfromtxt(FLAGS.data_dir + '/test_seq/boxing.binary_noisy.csv', delimiter=',')
    mocap_seq = np.genfromtxt(FLAGS.data_dir + '/test_seq/boxing.binary_our_result.csv', delimiter=',')

    #visualize(mocap_seq)
    animate(original_seq)
    #animate(mocap_seq,True)
    #animate(noisy_seq,True)

