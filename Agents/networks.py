from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config

import tensorflow as tf
import sympy as sp
import numpy as np
from sympy.utilities.lambdify import lambdify
from scipy.special import gamma as Gamma
import warnings

np_config.enable_numpy_behavior()
def suppress_runtime_warnings(func):
    """Decorator to suppress runtime warnings for a specific function."""

    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return func(*args, **kwargs)

    return wrapper
def generate_maxima_finder(degree, bounds):
    # Create a symbolic variable
    x = sp.Symbol('x')

    # Define coefficients as additional symbolic variables
    coefficients = [sp.Symbol(f'a{i}') for i in range(degree + 1)]

    # Construct the polynomial
    polynomial = sum([coefficients[i] * x**i for i in range(degree + 1)])

    # Compute the derivative
    derivative = sp.diff(polynomial, x)

    # Solve the derivative equals zero for critical points
    critical_points = sp.solve(derivative, x)
    # Pre-calculate the numerical functions for the critical points
    critical_point_funcs = [lambdify(
        coefficients,
        sp.simplify(cp),
        modules=[{"sqrt": np.lib.scimath.sqrt}, "numpy"]) for cp in critical_points
    ]
    @suppress_runtime_warnings
    def find_maxima(coeffs: np.array):
        if coeffs.shape[1] != degree + 1:
            raise ValueError(f'Expected {degree + 1} coefficients, got {len(coeffs)}')
        # split coeff array into individual column-vectors
        coef_vectors = np.split(coeffs.transpose(), coeffs.shape[1], axis=0)
        coef_vectors = [vector.flatten() for vector in coef_vectors]
        # Evaluate the critical points numerically
        critical_values_list = [cp_func(*coef_vectors) for cp_func in critical_point_funcs]
        critical_values = np.array(critical_values_list).transpose()
        critical_values[np.abs(critical_values.imag)>0.0005] = np.nan
        critical_values = np.real(critical_values)
        boundaries = np.tile(bounds, (critical_values.shape[0], 1))
        candidate_values = np.hstack((critical_values, boundaries))
        y_values = []
        for j in range(candidate_values.shape[1]):
            y_s = ((np.array([candidate_values[:,j]**i for i in range(coeffs.shape[1])]).transpose())*coeffs).sum(axis=1)
            y_values.append(y_s)

        extrema = np.array(y_values).transpose()
        extrema[(candidate_values < bounds[0]) | (candidate_values > bounds[1])] = np.nan
        # Find the pair with the maximum y-value
        max_values = np.nanmax(extrema, axis=1)
        mask_max = extrema == max_values[:, np.newaxis]
        indices = np.argmax(mask_max, axis=1)
        optima = candidate_values[np.arange(extrema.shape[0]), indices]
        # Return the x-value of the maxima and the corresponding max value
        return optima, max_values

    return find_maxima

def q_network_actor(
    lr: float = 0.0001,
    input_dims: tuple = (2,),
    output_dims = 3,
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
    model.add(Dense(output_dims, activation=None))
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    return model


def beta_moment(alpha, beta, n):
    """Calculate the nth moment of the Beta distribution."""
    n = tf.cast(n, dtype=alpha.dtype)  # Convert n to the same type as alpha and beta
    return tf.exp(
        tf.math.lgamma(alpha + n) + tf.math.lgamma(alpha + beta) - tf.math.lgamma(alpha) -
        tf.math.lgamma(alpha + beta + n)
    )



def expected_value_of_polynomial(poly_coefficients, beta_coefficients):
    """Compute the expected value of a polynomial where X follows a Beta distribution."""
    alpha = beta_coefficients[:,0]
    beta = beta_coefficients[:,1]

    # Number of coefficients in the polynomial
    num_coeffs = poly_coefficients.shape[1]

    # Calculating moments using TensorFlow operations
    moments = tf.stack([beta_moment(alpha, beta, n) for n in range(num_coeffs)], axis=1)

    return tf.reduce_sum(poly_coefficients * moments, axis=1)

def polynomial_loss(y_true, polynomial_coefficients):
    ev = y_true[:,0]
    betsize = y_true[:,1]

    # Create a range tensor for the polynomial degrees and cast to the same type as x
    degree_range = tf.cast(tf.range(polynomial_coefficients.shape[1]), dtype=betsize.dtype)

    # Calculate the powers of x
    x_powers = tf.pow(tf.expand_dims(betsize, axis=-1), degree_range)

    # Calculate predicted y-values
    y_pred = tf.reduce_sum(x_powers * polynomial_coefficients, axis=1)

    # Compute the loss (e.g., Mean Squared Error)
    return tf.reduce_mean(tf.square(ev - y_pred))
def q_continuous(
        input_dims: tuple = (2,),
        layer_sizes: tuple = (1024, 512, 256),
        dropout_rate = 0.2,
        degree = 4,
        lr=0.0001,
):
    model = Sequential()
    model.add(Input(shape=input_dims))
    for size in layer_sizes:
        model.add(Dense(size))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(dropout_rate))
    model.add(Dense(degree+1, activation=None))
    model.compile(optimizer=Adam(learning_rate=lr), loss=polynomial_loss)
    return model

def regret_loss(evs, action_probs, entropy_coef=1):
    # Convert evs to float32 for consistency
    evs = tf.cast(evs, tf.float32)
    # Compute the max Q-value across all actions
    max_q = tf.reduce_max(evs, axis=1)
    # Calculate the expected value under the policy
    ev_policy = tf.reduce_sum(action_probs * evs, axis=1)
    # Calculate the regret as the loss
    regret = tf.reduce_mean(max_q - ev_policy)
    entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)

    return regret - entropy_coef * tf.reduce_mean(entropy)


def betsize_regret(y_true, y_pred, entropy_coefficient=0.1):
    max_ev = y_true[:, 0]
    polynomials = y_true[:, 1:]

    # Assuming y_pred consists of beta_params
    beta_params = y_pred
    ev_policy = expected_value_of_polynomial(polynomials, beta_params)

    regret = tf.reduce_mean(max_ev - ev_policy)

    alpha = beta_params[:, 0]
    beta = beta_params[:, 1]
    dist = tfp.distributions.Beta(alpha, beta)
    entropy = dist.entropy()

    return regret - entropy_coefficient * tf.reduce_mean(entropy)


def bet_size_actor(input_shape=(2,), layer_sizes=(512, 256), dropout_rate=0.2, polynomial_degree=4, lr=0.0001):
    inputs = Input(shape=input_shape)
    x = inputs

    # Continuous branch for bet sizing - outputs alpha and beta parameters for the beta distribution
    for num_neurons in layer_sizes:
        x = Dense(num_neurons)(x)
        x = LeakyReLU()(x)
        x = Dropout(dropout_rate)(x)

    # Output the alpha and beta as softplus to ensure they are positive and suitable for beta distribution
    # Combine alpha and beta into a single output
    alpha = tf.clip_by_value(Dense(1, activation='softplus')(x), 1e-6, 1e6)
    beta = tf.clip_by_value(Dense(1, activation='softplus')(x), 1e-6, 1e6)

    combined_output = Concatenate()([alpha, beta])

    # Create the model
    model = Model(inputs=inputs, outputs=combined_output)

    # Compile the model with a single loss function
    model.compile(optimizer=Adam(learning_rate=lr), loss=betsize_regret)

    return model

def actor_with_regret(
    lr: float = 0.00001,
    input_dims: tuple = (2,),
    output_dims = 3,
    layer_sizes: tuple = (1024, 512, 256),
    dropout_rate: float = 0.2,
    loss: str = regret_loss,
):
    model = Sequential()
    model.add(Input(shape=input_dims))
    for size in layer_sizes:
        model.add(Dense(size))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dropout(dropout_rate))
    model.add(Dense(output_dims, activation="softmax"))
    model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    return model