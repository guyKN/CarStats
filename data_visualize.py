import numpy as np
import matplotlib.pyplot as plt

def graph_passing_line_times(passing_line_times: np.array, fps: int):
    time_dif = np.diff(passing_line_times, prepend=0)
    time_dif_seconds = time_dif/fps
    print(time_dif_seconds)
    plt.title("Histogram of difference between arrival times of cars")
    plt.xlabel("Time Between arrivals[seconds]")
    plt.ylabel("Probability Density")
    plt.hist(time_dif_seconds, density=True, bins=20)
    plt.show()