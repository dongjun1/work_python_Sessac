import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Step 1: Define a custom exception to signal stopping the animation
class AnimationStop(Exception):
    pass

# Step 2: Set up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 100)
line, = ax.plot(x, np.sin(x))

# Define a flag to control the animation
stop_flag = False
dt = 0

# Step 3: Define the update function
def update(frame):
    global stop_flag, dt
    if stop_flag or dt > 10:
        plt.close(fig)  # Close the figure window
        raise AnimationStop  # This will signal to stop the animation
    line.set_ydata(np.sin(x + frame / 10.0))
    dt += 1 
    return line,

# Step 4: Define a function to stop the animation
def stop_animation(event):
    global stop_flag
    stop_flag = True

# Connect the stop function to a key press event (e.g., pressing 'q')
# fig.canvas.mpl_connect('key_press_event', lambda event: stop_animation(event) if event.key == 'q' else None)

# Step 5: Run the animation and catch the custom exception
try:
    ani = animation.FuncAnimation(fig, update, frames=range(100), blit=True, repeat=False)
    plt.show()
except AnimationStop:
    print("Animation stopped by user.")

# Any additional code you want to run after the animation stops
print("This code runs after the animation stops.")