import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_swiss_roll_3D(data, iter=0):
    folder = 'save/figs'
    filename = f'iter_{iter}_3D.gif'
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='b', s=10)
    ax.set_title('3D Swiss Roll Dataset')

    def update(frame):
        ax.view_init(elev=10, azim=frame)

    ani = FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50)
    ani.save(file_path, writer='pillow')

    plt.close(fig)

def visualize_swiss_roll_2D(data, iter=0):
    folder = 'save/figs'
    filename = f'iter_{iter}_2D.png'
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], color='b', s=10)
    plt.title('2D Swiss Roll Projection')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    
    plt.savefig(file_path)
    plt.close()

def visualize_swiss_roll(data, iter):
    if data.shape[-1]==3:
        visualize_swiss_roll_3D(data, iter)
    else:
        visualize_swiss_roll_2D(data, iter)