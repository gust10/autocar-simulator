# live_plot.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from threading import Thread

# Buffers for live data
time_data = deque(maxlen=300)
speed_data = deque(maxlen=300)
target_data = deque(maxlen=300)

# Public method to add new data
def add_data(time_value, current_speed, target_speed):
    time_data.append(time_value)
    speed_data.append(current_speed)
    target_data.append(target_speed)

def start_plot():
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label="Current Speed")
    line2, = ax.plot([], [], label="Target Speed")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed")
    ax.legend()
    ax.grid(True)

    def update(frame):
        if len(time_data) == 0:
            return line1, line2
        ax.set_xlim(max(0, time_data[-1] - 10), time_data[-1])
        line1.set_data(time_data, speed_data)
        line2.set_data(time_data, target_data)
        return line1, line2

    ani = animation.FuncAnimation(fig, update, interval=100)
    plt.show()

def run_in_thread():
    thread = Thread(target=start_plot, daemon=True)
    thread.start()
