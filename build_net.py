import tensorflow as tf


def build_FCNN():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(500, activation='relu', input_shape=(500,)))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(2))
    return model


def main():
    model = build_FCNN()
    print(model.summary())


if __name__ == '__main__':
    main()