import numpy as np
import click


def load_data(single_path, double_path, file_name):
    """
    Takes traces from text data and loads them into a numpy array. Each row is a trace, first 500 columns are trace
    data, last 2 columns are time offsets. Saves the array.
    :param str single_path: Path to .txt file containing data from single pulses
    :param str double_path: Path to .txt file containing data from double pulses
    :param str file_name: Name for .npy file
    """
    # load the single pulses
    with open(single_path) as f:
        str = f.read()
    list = str.split('\n500')  # '\n500 is the start of each seperate trace
    num_pulses = len(list)
    data = np.zeros((502, num_pulses*2))  # *2 because will load double pulses into same array later
    for i in range(num_pulses):
        if i % 500 == 0:
            print(i)
        lines = list[i].split('\n')
        targets = lines[0].split()

        if i < 1:  # for some reason when the lines are split the first array has an extra element, so have to slice
            #  differently
            data[500, i] = float(targets[4])  # time offset for first pulse
            data[501, i] = float(targets[9])  # time offset for second pulse
        else:
            data[500, i] = float(targets[3])  # time offset for first pulse
            data[501, i] = float(targets[8])  # time offset for second pulse
        data[:500, i] = np.genfromtxt(lines, skip_header=1, usecols=(1))
    # load the double pulses
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

    data = np.transpose(data)
    np.save(file_name, data)


@click.command()
@click.argument('single_path', type=click.Path(exists=True, readable=True))
@click.argument('double_path', type=click.Path(exists=True, readable=True))
@click.argument('file_name', type=click.Path)
def main(single_path, double_path, file_name):
    load_data(single_path, double_path, file_name)


if __name__ == '__main__':
    main()
