import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
# import tensorflow.keras.layers as layers
# import tensorflow.keras.models as models

def ae(x_train, y_train, x_test, y_test):
    n_inputs = x_train.shape[1]
    n_outputs = 2
    n_latent = 50

    inputs = tf.keras.layers.Input(shape=(n_inputs, ))

    x = tf.keras.layers.Dense(100, activation='tanh')(inputs)
    latent = tf.keras.layers.Dense(n_latent, activation='tanh')(x)

    # Encoder
    encoder = tf.keras.models.Model(inputs, latent, name='encoder')
    encoder.summary()

    latent_inputs = tf.keras.layers.Input(shape=(n_latent, ))
    x = tf.keras.layers.Dense(100, activation='tanh')(latent_inputs)
    outputs = tf.keras.layers.Dense(n_inputs, activation='sigmoid')(x)

    # Decoder
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # 정상 데이터 만을 학습
    x_train_norm = x_train[y_train == 0]

    #autoencoder for training encoder & decoder
    autoencoder = tf.keras.models.Model(inputs, decoder(encoder(inputs)))
    autoencoder.compile(optimizer='adam', loss='mse')

    # training encoder
    autoencoder.fit(x_train_norm, x_train_norm, epochs=15, batch_size = 100, validation_data=(x_test, x_test))

    # result of encoder using x_train
    encoded = encoder.predict(x_train)

    # make classifier (model)
    classifier = tf.keras.Sequential([
        layers.Dense(32, input_dim=n_latent, activation='tanh'),
        layers.Dense(16, activation='relu'),
        layers.Dense(n_outputs, activation ='softmax')
    ])
    classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.summary()

    # classifier training using generated data(encoded)
    classifier.fit(encoded, y_train, batch_size=100, epochs=10)

    # Predict !
    pred_y = classifier.predict(encoder.predict(x_test)).argmax(axis=1)

    print(len(pred_y))
    y = y_test

    print(classification_report(y, pred_y))