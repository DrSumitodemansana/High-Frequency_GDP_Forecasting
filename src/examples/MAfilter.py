import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')


def MAfilter(data, window_size):
    """
    Moving average filter
    :param data: input data
    :param window_size: window size
    :return: filtered data
    """
    data = np.array(data)
    data_length = len(data)
    filtered_data = np.zeros(data_length)
    for i in range(data_length):
        if i < window_size:
            filtered_data[i] = np.mean(data[:i + 1])
        else:
            filtered_data[i] = np.mean(data[i - window_size:i + 1])  # Corregido aquí
    return filtered_data


if __name__ == "__main__":
    time = np.linspace(0, 10, 1000)
    data = np.sin(time) + np.random.normal(0, 0.1, 1000)
    filtered_data = MAfilter(data, 10)

    plt.plot(time, data, label='Datos originales')
    plt.plot(time, filtered_data, label='Datos filtrados', linewidth=2)
    plt.legend()
    plt.show()
