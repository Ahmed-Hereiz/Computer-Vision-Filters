from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.activations import LeakyReLU


def create_and_train_model(X_train, y_train, X_test, y_test):
    inputs = Input(shape=(96, 96, 1))
    x = BatchNormalization()(inputs)
    x = Conv2D(128, (3,3), padding="same", kernel_initializer=glorot_uniform(), activation=LeakyReLU(0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), padding="same", kernel_initializer=glorot_uniform(), activation=LeakyReLU(0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), padding="same", kernel_initializer=glorot_uniform(), activation=LeakyReLU(0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = BatchNormalization()(x)
    x = Conv2D(512, (3,3), padding="same", kernel_initializer=glorot_uniform(), activation=LeakyReLU(0.1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1028, kernel_initializer=glorot_uniform(), activation=LeakyReLU(0.1))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, kernel_initializer=glorot_uniform(), activation=LeakyReLU(0.1))(x)
    x = Dense(6, kernel_initializer=glorot_uniform())(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=7.1365e-06), metrics=['mean_squared_error'])

    checkpoint_filepath = 'landmark_detect_model.h5'

    checkpoint_callback = ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=200,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_callback]
    )

    return model, history
