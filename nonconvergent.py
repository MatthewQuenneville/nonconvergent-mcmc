import tensorflow_datasets as tfds
import tensorflow as tf
from functools import partial
import tinymcmc
from flax import linen as nn
import jax
import jax.numpy as jnp
from clu import metrics
from flax.training import train_state
from flax import struct
import optax
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt

def get_dataset(n_epochs, batch_size):
    """Load MNIST dataset

    Loads and generates batches from the MNIST dataset

    Parameters
    ----------
    n_epochs : int
        Number of epochs
    batch_size : int
        Number of images per batch

    Returns
    -------
    out : tf.data.Dataset 
        MNIST training dataset split into batches
    """
    
    dataset = tfds.load('mnist', split='train')
    dataset = dataset.map(lambda sample: {'image': 2 * (tf.cast(sample['image'], tf.float32) / 255. - 0.5)})
    dataset = dataset.repeat(n_epochs).shuffle(1024) 
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1) 

    return dataset

class CNN(nn.Module):
    """Convolutional Neural Network class"""

    @nn.compact
    def __call__(self, x):
        """Forward pass of CNN architecture

        Computes a forward pass of the CNN on a given image. Outputs 10 logits,
        which can be combined via Softmax for classification, or via LogSumExp
        for unsupervised generative training. 

        Parameters
        ----------
        x : ndarray
            Input data. Must be of shape (n_batch, 28, 28, 1).

        Returns
        -------
        out : ndarray
            Output logits of shape (n_batch, 10)
        """
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        if len(x.shape)==4:
            x = x.reshape((-1, 3136))
        else:
            x = x.reshape((3136,))

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)

        x = nn.Dense(features=10)(x)
        return x

def create_train_state(module, key, learning_rate):
    """Create a train state object
    
    Creates a train state object for training

    Parameters
    ----------
    module : flax.linen.module
        Flax Linen neural network object
    key : jax.prng.PRNGKeyArray
        Random key
    learning_rate : float
        Learning rate parameter
 
    Returns
    -------
    out : flax.training.train_state.TrainState
        Output training state
    """
    
    params = module.init(key, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.sgd(learning_rate)
    return train_state.TrainState.create(
        apply_fn=module.apply, params=params, tx=tx)

@jax.jit
def generative_train_step(state, batch, samples):
    """Unsupervised generative training step

    Performs a single unsupervised generative training step
    
    Parameters
    ----------
    state : flax.training.train_state.TrainState
        Current training state
    batch : ndarray
        Input batch of data images. 
        Should be of shape (n_batch, 28, 28, 1).
    samples : ndarray
        Input batch of sampled images. 
        Should be of shape (n_batch, 28, 28, 1).

    Returns
    -------
    out : flax.training.train_state.TrainState
        Updated training state
    """
    
    def loss_fn(params):
        batch_logits = state.apply_fn({'params': params}, batch)
        sample_logits = state.apply_fn({'params': params}, samples)
        energy_loss = -jax.numpy.mean(jax.scipy.special.logsumexp(-batch_logits, axis=-1) \
                                      - jax.scipy.special.logsumexp(-sample_logits, axis=-1))
        return energy_loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state

@partial(jax.jit, static_argnames=('n_steps'))
def sampling_step(state, samples, key, step_size=0.015, n_steps=100):
    """MCMC Sampling step

    Performs a set number of MALA sampling steps
    
    Parameters
    ----------
    state : flax.training.train_state.TrainState
        Current training state
    samples : ndarray
        Input batch of sampled images.
        Should be of shape (n_batch, 28, 28, 1).
    key : jax.prng.PRNGKeyArray
        Random key
    step_size : float, optional
        Step-size argument for MALA sampler.
        Defaults to 0.015.
    n_steps : int, optional
        Number of sampling steps to perform.
        Defaults to 100.
 
    Returns
    -------
    out : ndarray
        Output batch of sampled images.
    """
    samples = samples
    E_func = lambda x: -jax.scipy.special.logsumexp(-state.apply_fn({'params': state.params}, x),axis=-1)
    for i in range(n_steps):
        key, key_step = jax.random.split(key, 2)
        samples = tinymcmc.step_mala(
            key_step, E_func, samples, step_size, 
            metropolize=False, noise_scale=0.0)
    return samples


# Define training dataset
n_epochs = 100
batch_size = 100
learning_rate = 0.0001
sigma_noise = 0.03 # Added noise to stabilize training

dataset = get_dataset(n_epochs, batch_size)
n_steps_per_epoch = dataset.cardinality().numpy() // n_epochs

# Initialize CNN model
key = jax.random.key(0)
key, key_init = jax.random.split(key)
cnn = CNN()
state = create_train_state(cnn, key_init, learning_rate)

# Initialize Orbax checkpointer
checkpointer = ocp.StandardCheckpointer()
path = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')

# Perform training loop
for step,batch in enumerate(dataset.as_numpy_iterator()):

    key, key_init, key_noise, key_sampling = jax.random.split(key, 4)
    samples = 2*(jax.random.uniform(key_init, batch['image'].shape)-0.5)
    batch = batch['image']+sigma_noise*jax.random.normal(key_noise, batch['image'].shape)
    samples = sampling_step(state, samples, key_sampling)
    state = generative_train_step(state, batch, samples)
  
    if (step+1) % n_steps_per_epoch == 0:
        checkpointer.save(path / f'checkpoint_{(step+1) // n_steps_per_epoch}', state.params)
    
data = batch.reshape((-1,10,28,28))
data = jnp.einsum('ijkl->ikjl',data)
data = data.reshape((-1,280))
samples = samples.reshape((-1, 10, 28, 28))
samples = jnp.einsum('ijkl->ikjl',samples)
samples = samples.reshape((-1,280))

fig, axes = plt.subplots(1,2,sharex=True,sharey=True, figsize=(12,6))
axes[0].imshow(data, cmap='Greys_r', vmin=-1, vmax=1)
axes[0].set_title('Data samples')
axes[1].imshow(samples,cmap='Greys_r', vmin=-1, vmax=1)
axes[1].set_title('Generated samples')
axes[0].get_xaxis().set_visible(False)
axes[0].get_yaxis().set_visible(False)
axes[1].get_xaxis().set_visible(False)
axes[1].get_yaxis().set_visible(False)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
plt.suptitle('Non-convergent MCMC')
plt.show()
