from tensorflow import keras
from tensorflow.keras import activations, layers, regularizers


class Subencoder(layers.Layer):
	def __init__(
			self,
			conv_padding: str = 'causal',
			pool_padding: str = 'valid',
			**kwargs):
		super(Subencoder, self).__init__(**kwargs)
		self.conv = layers.Conv1D(padding=conv_padding, **kwargs)
		self.pool = layers.MaxPooling1D(padding=pool_padding, **kwargs)

	def call(self, inputs, training=None, **kwargs):
		if training:
			source, target = inputs
			source, target = self.conv(source), self.conv(target)
			pooled = (self.pool(source), self.pool(target))
		else:
			target = inputs
			target = self.conv(target)
			pooled = self.pool(target)
		return target, pooled


class Encoder(layers.Layer):
	def __init__(
			self,
			filters: int = 90,
			kernel_size: int = 10,
			pool_size: int = 5,
			conv_padding: str = 'causal',
			pool_padding: str = 'valid',
			activation=activations.relu,
			kernel_initializer=keras.initializers.glorot_uniform,
			kernel_regularizer=regularizers.L2,
			bias_regularizer=regularizers.L2,
			activity_regularizer=regularizers.L2,
			noise_stddev: float = 0.1,
			**kwargs):
		super(Encoder, self).__init__(**kwargs)
		self.noise = layers.GaussianNoise(stddev=noise_stddev, **kwargs)
		encoder_kwargs = {
			'filters': filters,
			'kernel_size': kernel_size,
			'pool_size': pool_size,
			'conv_padding': conv_padding,
			'pool_padding': pool_padding,
			'activation': activation,
			'kernel_initializer': kernel_initializer,
			'kernel_regularizer': kernel_regularizer,
			'bias_regularizer': bias_regularizer,
			'activity_regularizer': activity_regularizer}
		self.encode1 = Subencoder(**encoder_kwargs, **kwargs)
		self.encode2 = Subencoder(**encoder_kwargs, **kwargs)

	def call(self, inputs, training=None, **kwargs):
		if training:
			source, target = inputs
			inputs = (self.noise(source), self.noise(target))
			target, pooled = self.encode1(inputs, training=training)
			target, pooled = self.encode2(pooled, training=training)
		else:
			target = inputs
			target, pooled = self.encode1(target, training=training)
			target, pooled = self.encode2(pooled, training=training)
		return target, pooled


class Subdecoder(layers.Layer):
	def __init__(
			self,
			filters: int = 90,
			**kwargs):
		super(Subdecoder, self).__init__(**kwargs)
		self.conv = layers.Conv1D(filters=filters, **kwargs)
		self.cat = layers.Concatenate()
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
