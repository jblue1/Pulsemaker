import numpy as np
import pandas as pd


def load_timeoffset_data(single_data_path, double_data_path, num_pulses, file_name):
    """
    Takes traces from text data and loads them into a numpy array. Each row is a trace, first 500 columns are trace
     data, last 2 columns are time offsets. Saves the array.
    :param str single_data_path: Path to .txt file containing data from single pulses
    :param str double_data_path: Path to .txt file containing data from double pulses
    :param int num_pulses: Number of pulses to load
    :param str file_name: Name for .npy file
    """
    df1 = pd.read_csv(single_data_path, delimiter=' ', header=None,
                      names=['nsamples', 'A1', 'K1', 'K2', 'X1', 'C', 'A2', 'K3', 'K4', 'X2'], index_col=False)
    df2 = pd.read_csv(double_data_path, delimiter=' ', header=None,
                      names=['nsamples', 'A1', 'K1', 'K2', 'X1', 'C', 'A2', 'K3', 'K4', 'X2'], index_col=False)

    df = df1.append(df2, ignore_index=True)
    meanA = np.mean(df['A1'])
    stdA = np.std(df['A1'])
    maxX1 = np.max(df['X1'])
    maxX2 = np.max(df['X2'])
    trace_data = np.zeros((502, num_pulses))
    for i in range(num_pulses):
        if i % 100 == 0:
            print(i)
        index1 = i * 500 + i + 1
        index2 = index1 + 500
        trace_data[:500, i:i + 1] = (df[['A1']][index1:index2] - meanA) / stdA
        targets = np.transpose(np.array(df[['X1', 'X2']][index1 - 1:index1]))
        trace_data[500, i:i + 1] = (targets[0]) / maxX1
        trace_data[501, i:i + 1] = targets[1] / maxX2

    trace_data = np.transpose(trace_data)
    np.random.shuffle(trace_data)  # shuffle data along first axis

    np.save(file_name, trace_data)


def main():
    load_timeoffset_data('~/ML-Research/Pulsemaker/pulsemaker/pulsemaker/singles_10k_0_1.txt',
                         '~/ML-Research/Pulsemaker/pulsemaker/pulsemaker/doubles_10k_0_1.txt',
                         20000,
                         '20k_singles_doubles_0_1')


if __name__ == '__main__':
    main()
