import numpy as np


def load_by_string(single_path, double_path, file_name):
    """
    Takes traces from text data and loads them into a numpy array. Each row is a trace, first 500 columns are trace
    data, last 2 columns are time offsets. Saves the array.
    :param str single_path: Path to .txt file containing data from single pulses
    :param str double_path: Path to .txt file containing data from double pulses
    :param str file_name: Name for .npy file
    """
    with open(single_path) as f:
        str = f.read()
    list = str.split('\n500')
    num_pulses = len(list)
    data = np.zeros((502, num_pulses*2))
    for i in range(num_pulses):
        if i % 500 == 0:
            print(i)
        lines = list[i].split('\n')
        targets = lines[0].split()
        if i < 1:
            data[500, i] = float(targets[4])
            data[501, i] = float(targets[9])
        else:
            data[500, i] = float(targets[3])
            data[501, i] = float(targets[8])
        data[:500, i] = np.genfromtxt(lines, skip_header=1, usecols=(1))
    with open(double_path) as f:
         str = f.read()
    list = str.split('\n500')
    for i in range(num_pulses):
        if i % 500 == 0:
            print(i)
        lines = list[i].split('\n')
        targets = lines[0].split()
        if i < 1:
            data[500, i + num_pulses] = float(targets[4])
            data[501, i + num_pulses] = float(targets[9])
        else:
            data[500, i + num_pulses] = float(targets[3])
            data[501, i + num_pulses] = float(targets[8])
        data[:500, i + num_pulses] = np.genfromtxt(lines, skip_header=1, usecols=(1))

    print(data.shape)
    data = np.transpose(data)
    print(data.shape)
    print(data.mean(0).shape)
    np.save(file_name, data)


def main():
    load_by_string('singles_100k_0_1.txt',
                   'doubles_100k_0_1.txt',
                   '200k_singles_doubles_0')
    #load_timeoffset_data('singles_100k_0_1.txt',
                        # 'doubles_100k_0_1.txt',
                        # 200000,
                         #'200k_singles_doubles_0_1')


if __name__ == '__main__':
    main()
