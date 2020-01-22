import tensorflow as tf


def energy_model():
    inputDense = tf.keras.Input(shape=(500,))
    inputConv = tf.keras.Input(shape=(500, 1,))

    x = tf.keras.layers.Dense(500, activation='relu')(inputDense)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dense(200, activation='relu')(x)

    y = tf.keras.layers.Conv1D(64, 5, activation='relu')(inputConv)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Conv1D(64, 5, activation='relu')(y)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Conv1D(64, 5, activation='relu')(y)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Conv1D(64, 5, activation='relu')(y)
    y = tf.keras.layers.Flatten()(y)

    concat = tf.keras.layers.concatenate([x, y])
    out = tf.keras.layers.Dense(500, activation='relu')(concat)
    out = tf.keras.layers.Dense(500, activation='relu')(out)
    out = tf.keras.layers.Dropout(.3)(out)
    out = tf.keras.layers.Dense(200, activation='relu')(out)

    out = tf.keras.layers.Dense(2)(out)

    return tf.keras.Model([inputDense, inputConv], out)

def build_FCNN():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(500, activation='relu', input_shape=(500,)))
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(2))
    return model


def build_Conv1D():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(64, 20, activation='relu', input_shape=(500, 1, )))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Conv1D(64, 20, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Conv1D(64, 20, activation='relu'))
    model.add(tf.keras.layers.Conv1D(64, 20, activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(2))

    return model


def time_model():
    inputDense = tf.keras.Input(shape=(500,))
    inputConv = tf.keras.Input(shape=(500, 1,))

    x = tf.keras.layers.Dense(500, activation='relu')(inputDense)
    x = tf.keras.layers.Dense(500, activation='relu')(x)
    x = tf.keras.layers.Dense(500, activation='relu')(x)

    y = tf.keras.layers.Conv1D(64, 20, activation='relu')(inputConv)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Conv1D(64, 20, activation='relu')(y)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Conv1D(64, 20, activation='relu')(y)
    y = tf.keras.layers.Conv1D(64, 20, activation='relu')(y)
    y = tf.keras.layers.MaxPooling1D(2)(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(500)(y)

    concat = tf.keras.layers.concatenate([x, y])
    out = tf.keras.layers.Dense(500, activation='relu')(concat)
    out = tf.keras.layers.Dense(250, activation='relu')(out)
    out = tf.keras.layers.Dense(2)(out)

    return tf.keras.Model([inputDense, inputConv], out)


def main():
    model = energy_model()
    print(model.summary())


if __name__ == '__main__':
    main()
