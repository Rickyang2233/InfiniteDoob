import jax.numpy as jnp
from flax import linen as nn

class ScoreMLP(nn.Module):
    """ Multi-layer perceptron """
    output_dim: int
    hidden_dims: list
    activation: str='tanh'

    def setup(self):
        if self.activation == 'tanh':
            act = nn.tanh
        elif self.activation == 'relu':
            act = nn.relu
        elif self.activation == 'sigmoid':
            act = nn.sigmoid
        elif self.activation == 'elu':
            act = nn.elu
        else:
            raise NotImplementedError
        
        layers = []
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Dense(hidden_dim))
            layers.append(act)
        layers.append(nn.Dense(self.output_dim))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    