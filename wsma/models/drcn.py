import numbers
from typing import Callable, Union

from tensorflow import keras
from tensorflow.keras import activations, layers, regularizers

Activation = Union[Callable[..., numbers.Real], str]
Regularizer = Union[Callable[..., numbers.Real], str]


# TODO Some hyper-parameters not specified by the paper
#   L2 factor
#   Recurrent cell type

class Subencoder(layers.Layer):
	def __init__(
			self,
			*,
			filters: int = 90,
			pool_size: int = 5,
			activation: Activation = activations.relu,
			bias_regularizer: Regularizer = regularizers.L2,
			kernel_regularizer: Regularizer = regularizers.L2,
			**kwargs):
		super(Subencoder, self).__init__(**kwargs)
		self.conv = layers.Conv1D(
			filters=filters,
			activation=activation,
			bias_regularizer=bias_regularizer,
			kernel_regularizer=kernel_regularizer,
			**kwargs)
		self.pool = layers.MaxPooling1D(pool_size=pool_size, **kwargs)

	def call(self, inputs, training=None, **kwargs):
		if training:
			source, target = inputs
			src_conv, tar_conv = (self.conv(source), self.conv(target))
			pooled = (self.pool(src_conv), self.pool(tar_conv))
		else:
			tar_conv = self.conv(inputs)
			pooled = self.pool(tar_conv)
		return tar_conv, pooled


class Encoder(layers.Layer):
	def __init__(
			self,
			*,
			stddev: numbers.Real = 0.1,
			filters: int = 90,
			pool_size: int = 5,
			activation: Activation = activations.relu,
			bias_regularizer: Regularizer = regularizers.L2,
			kernel_regularizer: Regularizer = regularizers.L2,
			**kwargs):
		super(Encoder, self).__init__(**kwargs)
		self.noise = layers.GaussianNoise(stddev=stddev, **kwargs)
		self.encode1 = Subencoder(
			filters=filters,
			pool_size=pool_size,
			activation=activation,
			bias_regularizer=bias_regularizer,
			kernel_regularizer=kernel_regularizer,
			**kwargs)
		self.encode2 = Subencoder(
			filters=filters,
			pool_size=pool_size,
			activation=activation,
			bias_regularizer=bias_regularizer,
			kernel_regularizer=kernel_regularizer,
			**kwargs)

	def call(self, inputs, training=None, **kwargs):
		source, target = inputs
		fuzzy = (self.noise(source), self.noise(target))
		target1, pooled = self.encode1(fuzzy, training=training)
		target2, pooled = self.encode2(pooled, training=training)
		return (target1, target2), pooled


class Subdecoder(layers.Layer):
	def __init__(
			self,
			filters: int = 90,
			**kwargs):
		super(Subdecoder, self).__init__(**kwargs)
		self.conv = layers.Conv1D(filters=filters, **kwargs)
		self.cat = layers.Concatenate()
		# TODO Should be upsampled at the same rate as downsampled by encoder
		self.up_sample = layers.UpSampling1D()

	def call(self, inputs, **kwargs):
		encoded, convolved = inputs
		convolved = self.conv(convolved)
		concatenated = self.cat(convolved, encoded)
		return self.up_sample(concatenated)


class Decoder(layers.Layer):
	def __init__(self, filters: int = 90, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		self.decode1 = Subdecoder(filters=filters, **kwargs)
		self.decode2 = Subdecoder(filters=filters, **kwargs)
		self.conv = layers.Conv1D(filters=filters, **kwargs)

	def call(self, inputs, **kwargs):
		(target1, target2), pooled = inputs
		decoded1 = self.decode1((pooled, target1))
		decoded2 = self.decode1((decoded1, target2))
		return self.conv(decoded2)


class Classifier(layers.Layer):
	def __init__(
			self,
			units: int = 80,
			*,
			activation: Activation = activations.relu,
			**kwargs):
		super(Classifier, self).__init__(**kwargs)
		rnn = layers.GRU(units=units, activation=activation)
		self.rnn = layers.Bidirectional(rnn)
		# TODO Change to softmax if multi-class, instead of binary
		self.classify = layers.Activation(activations.sigmoid)

	def call(self, inputs, **kwargs):
		seq = self.rnn(inputs)
		return self.classify(seq)


class DRCN(keras.Model):

	def __init__(self, **kwargs):
		super(DRCN, self).__init__(**kwargs)
		self.encode = Encoder(**kwargs)
		self.decode = Decoder(**kwargs)
		self.classify = Classifier(**kwargs)

	def call(self, inputs, **kwargs):
		encoded = self.encode(inputs)
		decoded = self.decode(encoded)
		classified = self.classify(encoded)
		return classified, decoded

	def get_config(self):
		return super(DRCN, self).get_config()
