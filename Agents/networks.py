from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def actor_network(input_shape=(52,), shared_layers=(512, 256),
                   categorical_branch_layers=(256, 128, 64), continuous_branch_layers=(128, 64),
                   dropout_rate=0.2, num_classes=3):
    # Input layer
    inputs = Input(shape=input_shape)

    # Shared layers
    x = inputs
    for num_neurons in shared_layers:
        x = Dense(num_neurons)(x)
        x = LeakyReLU()(x)
        x = Dropout(dropout_rate)(x)

    # Categorical branch for choosing action type
    categorical_x = x
    for num_neurons in categorical_branch_layers:
        categorical_x = Dense(num_neurons)(categorical_x)
        categorical_x = LeakyReLU()(categorical_x)
        categorical_x = Dropout(dropout_rate)(categorical_x)
    categorical_output = Dense(num_classes, activation='softmax', name='categorical_output')(categorical_x)

    # Continuous branch for bet sizing - outputs alpha and beta parameters for the beta distribution
    continuous_x = x
    for num_neurons in continuous_branch_layers:
        continuous_x = Dense(num_neurons)(continuous_x)
        continuous_x = LeakyReLU()(continuous_x)
        continuous_x = Dropout(dropout_rate)(continuous_x)
    # Output the alpha and beta as softplus to ensure they are positive and suitable for beta distribution
    alpha_output = Dense(1, activation='softplus', name='alpha_output')(continuous_x)
    beta_output = Dense(1, activation='softplus', name='beta_output')(continuous_x)

    # Create model
    model = Model(inputs=inputs, outputs=[categorical_output, alpha_output, beta_output])

    # Compile the model with Adam optimizer and custom loss function
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'categorical_output': 'categorical_crossentropy',
                        'alpha_output': 'mse',
                        'beta_output': 'mse'},
                  metrics={'categorical_output': 'accuracy'})

    return model

def critic_network(
    lr: float = 0.00001,
    input_dims: tuple = (2,),
    layer_sizes: tuple = (1024, 512, 256),
    dropout_rate: float = 0.2,
    loss: str = "MSE",

):
    model = Sequential()
    model.add(Input(shape=input_dims))
    for size in layer_sizes:
        model.add(Dense(size))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation=None))
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    return model