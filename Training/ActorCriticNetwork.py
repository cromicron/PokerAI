import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense 

class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, hidden_layers = (1024,512),
            name='actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.n_neurons = []
        for layer in hidden_layers:
            self.n_neurons.append(layer)
        self.n_actions = n_actions
        self.model_name = name
        self.hidden_layers = []
        for layer in self.n_neurons:            
            self.hidden_layers.append(Dense(layer, activation='LeakyReLU'))

        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.hidden_layers[0](state)
        for i in range(1,len(self.hidden_layers)):
            value = self.hidden_layers[i](value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi