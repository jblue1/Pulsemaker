import numpy as np
import functions
import click


def load_data(single_path, double_path, file_name):
    """
    Takes traces from text data and loads them into a numpy array. Each row is a trace, first 500 columns are trace
    data, 501 and 502 are time offsets, and 503 and 504 are pulse energies. Saves the array.
    :param str single_path: Path to .txt file containing data from single pulses
    :param str double_path: Path to .txt file containing data from double pulses
    :param str file_name: Name for .npy file
    """
    # load the single pulses
    with open(single_path) as f:
        str = f.read()
    list = str.split('\n500')  # '\n500 is the start of each separate trace
    num_pulses = len(list)
    data = np.zeros((504, num_pulses*2))  # *2 because will load double pulses into same array later
    for i in range(num_pulses):
        if i % 500 == 0:
            print(i)
        lines = list[i].split('\n')
        targets = lines[0].split()

        # get time offsets and pulse energies from each pulse from header
        # header ordered as nsamples A1 K1 K2 X1 C A2 K3 K4 X2
        if i < 1:  # when lines are split only the first array gets the nsamples, so have to slice differently
            # have to cast from string to float
            A1 = float(targets[1])
            k1 = float(targets[2])
            k2 = float(targets[3])
            x1 = float(targets[4])
            assert x1 > 0
            data[500, i] = x1  # time offset for first pulse
            # single pulse, so  second time offset is 0
            data[502, i] = functions.pulseAmplitude(A1, k1, k2, x1)  # first pulse energy
            # single pulse, so second pulse energy is zero
        else:
            A1 = float(targets[0])
            k1 = float(targets[1])
            k2 = float(targets[2])
            x1 = float(targets[3])
            assert x1 > 0
            data[500, i] = x1  # time offset for first pulse
            # single pulse, so  second time offset is 0
            data[502, i] = functions.pulseAmplitude(A1, k1, k2, x1)  # first pulse energy
            assert data[502, i] > 0
            # single pulse, so second pulse energy is zero
        data[:500, i] = np.genfromtxt(lines, skip_header=1, usecols=(1))

    # load the double pulses
    with open(double_path) as f:
         str = f.read()
    list = str.split('\n500')
    for j in range(num_pulses):
        if j % 500 == 0:
            print(j)
        lines = list[j].split('\n')
        targets = lines[0].split()
        if j < 1:
            # have to cast from string to float
            A1 = float(targets[1])
            k1 = float(targets[2])
            k2 = float(targets[3])
            x1 = float(targets[4])
            A2 = float(targets[6])
            k3 = float(targets[7])
            k4 = float(targets[8])
            x2 = float(targets[9])
            assert x1 > 0
            assert x2 > 0
            assert x1 < x2
            data[500, j + num_pulses] = x1  # time offset for first pulse
            data[501, j + num_pulses] = x2  # time offset for second pulse
            data[502, j + num_pulses] = functions.pulseAmplitude(A1, k1, k2, x1)  # first pulse energy
            data[503, j + num_pulses] = functions.pulseAmplitude(A2, k3, k4, x2)  # second pulse energy
            assert data[502, j + num_pulses] > 0
            assert data[503, j + num_pulses] > 0

        else:
            A1 = float(targets[0])
            k1 = float(targets[1])
            k2 = float(targets[2])
            x1 = float(targets[3])
            A2 = float(targets[5])
            k3 = float(targets[6])
            k4 = float(targets[7])
            x2 = float(targets[8])
            assert x1 > 0
            assert x2 > 0
            assert x1 < x2
            data[500, j + num_pulses] = x1  # time offset for first pulse
            data[501, j + num_pulses] = x2  # time offset for second pulse
            data[502, j + num_pulses] = functions.pulseAmplitude(A1, k1, k2, x1)  # first pulse energy
            data[503, j + num_pulses] = functions.pulseAmplitude(A2, k3, k4, x2)  # second pulse energy
            assert data[502, j + num_pulses] > 0
            assert data[503, j + num_pulses] > 0

        data[:500, j + num_pulses] = np.genfromtxt(lines, skip_header=1, usecols=(1))

    data = np.transpose(data)
    np.random.shuffle(data)
    np.save(file_name, data)


@click.command()
@click.argument('single_path', type=click.Path(exists=True, readable=True))
@click.argument('double_path', type=click.Path(exists=True, readable=True))
@click.argument('file_name', type=click.Path(exists=False))
def main(single_path, double_path, file_name):
    load_data(single_path, double_path, file_name)


if __name__ == '__main__':
    main()
