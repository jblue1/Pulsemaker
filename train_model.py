import tensorflow as tf
import matplotlib.pyplot as plt
import os
import click
from datetime import date
import build_net
import pandas as pd
from contextlib import redirect_stdout
import numpy as np
from tensorflow.keras.callbacks import Callback


class Metrics(Callback):
    def __init__(self, val_features, val_features_conv, val_targets):
        self.val_features = val_features
        self.val_features_conv = val_features_conv
        self.val_targ = val_targets

    def on_train_begin(self, logs={}):
        self.val_percent_errors = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict([self.val_features, self.val_features_conv])))

        error = np.abs(self.val_targ[:, 0] - val_predict[:, 0])
        _val_percent_error = np.average((error / self.val_targ[:, 0]) * 100)
        self.val_percent_errors.append(_val_percent_error)

        print(' - validation single pulse percent error: {} % '.format(_val_percent_error))
        print(' ')
        return


def plot_loss(loss, val_loss, mae, save_dir):
    """
    Makes a plot of training and validation loss over the course of training.
    :param loss: training loss
    :param val_loss: validation loss
    :param mae: Mean Absolute Error
    :param save_dir: directory to save the image to
    """
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='Training Loss (MSE)')
    plt.plot(epochs, val_loss, label='Validation Loss (MSE)')
    plt.plot(epochs, mae, label='MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training/Validation Loss and MAE')
    plt.legend()
    filename = os.path.join(save_dir, 'Loss_history.png')
    plt.savefig(filename)
    plt.close()


def write_info_file(save_dir, data_path, batch_size, epochs, lr, run_number, times):
    """
    Writes a text file to the save directory with a summary of the hyper-parameters used for training
    :param str save_dir: path to directory to save the file to
    :param str data_path: path to .h5 data file
    :param int batch_size: size of batches used in training
    :param int epochs: number of epochs network was trained for
    :param float lr: learning rate for the optimizer
    :param str run_number: Run number of the day
    :param bool times: Was the model trained on time (true) or energies (false)
    """
    filename = os.path.join(save_dir, 'run_info.txt')
    info_list = ['ContextEncoder Hyper-parameters: Run {} \n'.format(run_number),
                 'Training data found at: {} \n'.format(data_path),
                 'Batch Size: {} \n'.format(batch_size),
                 'Epochs: {} \n'.format(epochs),
                 'Learning Rate: {} \n'.format(lr),
                 'Times: {} \n'.format(times)]

    with open(filename, 'w') as f:
        f.writelines(info_list)


@click.command()
@click.argument('data_path', type=click.Path(exists=True, readable=True))
@click.option('--batch_size', default=32)
@click.option('--num_pulses', default=-1)
@click.option('--epochs', default=50)
@click.option('--lr', default=0.0001, help='Learning rate for Adam optimizer')
@click.option('--run_number', default=1, help='ith run of the day')
@click.option('--times/--energies', default=True)
def main(data_path, batch_size, num_pulses, epochs, lr, run_number, times):
    today = str(date.today())
    run_number = '_' + str(run_number)
    save_dir = './Run_' + today + run_number

    if os.path.exists(save_dir):
        ans = input(
            'The directory this run will write to already exists, would you like to overwrite it? ([y/n])')
        if ans == 'y':
            pass
        else:
            return
    else:
        os.makedirs(save_dir)
    write_info_file(save_dir, data_path, batch_size, epochs, lr, run_number, times)

    if times:
        model = build_net.time_model()
    else:
        model = build_net.energy_model()
    # write .txt file with model summary
    filename = os.path.join(save_dir, 'modelsummary.txt')
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            model.summary()

    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    # load and normalize data
    data = np.load(data_path)
    print('Loaded Data')
    features = data[:, :500]
    features_conv = data[:, :500]
    if times:
        targets = data[:, 500:502]
        print(targets.max(axis=0))
        targets /= targets.max(axis=0)
    else:
        print('Loading energies as target data')
        targets = data[:, 502:]
        targets /= np.std(targets, axis=0)
    features_conv = features_conv / np.max(features_conv)
    features_conv = tf.expand_dims(features_conv, -1)
    features = features / np.max(features)


    split = round(0.8 * len(features))
    train_features = features[:split]
    val_features = features[split:]

    train_targets = targets[:split]
    val_targets = targets[split:]

    train_features_conv = features_conv[:split]
    val_features_conv = features_conv[split:]

    # set up checkpoints
    checkpoint_path = os.path.join(save_dir, "checkpoints/cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)
    metrics = Metrics(val_features, val_features_conv, val_targets)
    history = model.fit([train_features, train_features_conv],
                        train_targets,
                        validation_data=([val_features, val_features_conv], val_targets),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[cp_callback, metrics])

    # save losses in dataframe
    loss = pd.Series(history.history['loss'])
    val_loss = pd.Series(history.history['val_loss'])
    mae = pd.Series(history.history['val_mae'])
    loss_df = pd.DataFrame({'Training Loss': loss,
                            'Val Loss': val_loss,
                            'MAE': mae})
    filename = os.path.join(save_dir, 'losses.csv')
    loss_df.to_csv(filename)  # save losses for further plotting/analysis
    plot_loss(loss, val_loss, mae, save_dir)


if __name__ == '__main__':
    main()
