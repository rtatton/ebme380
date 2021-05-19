from tensorflow import keras
from tensorflow.keras import activations, layers, regularizers

L2 = regularizers.l2()
SAMPLING_RATE = 5


def create_conv1d():
	return layers.Conv1D(
		filters=90,
		kernel_size=10,
		activation=activations.relu,
		kernel_regularizer=L2,
		bias_regularizer=L2,
		activity_regularizer=L2)


class _Subencoder(layers.Layer):
	def __init__(self, **kwargs):
		super(_Subencoder, self).__init__(**kwargs)
		self.conv = create_conv1d()
		self.pool = layers.MaxPooling1D(pool_size=SAMPLING_RATE)

	def call(self, inputs, training=None, **kwargs):
		if training:
			source, target = inputs
			source, convolved = self.conv(source), self.conv(target)
			pooled = (self.pool(source), self.pool(convolved))
			result = (pooled, convolved)
		else:
			convolved = self.conv(inputs)
			result = self.pool(convolved)
		return result


class _Encoder(layers.Layer):
	def __init__(self, **kwargs):
		super(_Encoder, self).__init__(**kwargs)
		self.noise = layers.GaussianNoise(stddev=0.1)
		self.encode1 = _Subencoder(**kwargs)
		self.encode2 = _Subencoder(**kwargs)

	def call(self, inputs, training=None, **kwargs):
		if training:
			source, target = inputs
			inputs = (self.noise(source), self.noise(target))
			pooled, convolved1 = self.encode1(inputs, training=training)
			pooled, convolved2 = self.encode2(pooled, training=training)
			result = (pooled, (convolved1, convolved2))
		else:
			pooled = self.encode1(inputs, training=training)
			result = self.encode2(pooled, training=training)
		return result


class _Subdecoder(layers.Layer):
	def __init__(self, **kwargs):
		super(_Subdecoder, self).__init__(**kwargs)
		self.conv = create_conv1d()
		self.cat = layers.Concatenate()
		self.up_sample = layers.UpSampling1D(size=SAMPLING_RATE)

	def call(self, inputs, training=None, **kwargs):
		if training:
			pooled_or_up_sampled, convolved = inputs
			convolved = self.conv(pooled_or_up_sampled)
			concatenated = self.cat(pooled_or_up_sampled, convolved)
			return self.up_sample(concatenated)


class _Decoder(layers.Layer):
	def __init__(self, **kwargs):
		super(_Decoder, self).__init__(**kwargs)
		self.decode1 = _Subdecoder(**kwargs)
		self.decode2 = _Subdecoder(**kwargs)
		self.conv = create_conv1d()

	def call(self, inputs, training=None, **kwargs):
		if training:
			pooled, (convolved1, convolved2) = inputs
			up_sampled = self.decode1((pooled, convolved1))
			up_sampled = self.decode2((up_sampled, convolved2))
			return self.conv(up_sampled)


class _Classifier(layers.Layer):
	def __init__(self, **kwargs):
		super(_Classifier, self).__init__(**kwargs)
		self.rnn = layers.Bidirectional(layers.GRU(
			units=80,
			kernel_regularizer=L2,
			recurrent_regularizer=L2,
			activity_regularizer=L2))
		self.classify = layers.Activation(activations.sigmoid)

	def call(self, inputs, **kwargs):
		sequence = self.rnn(inputs)
		return self.classify(sequence)


class DRCN(keras.Model):
	"""Deep reconstruction classification network.

	References:
		A. Saeed, T. Ozcelebi, J. Lukkien, J. B. F. van Erp and S. Trajanovski,
		"Model Adaptation and Personalization for Physiological Stress
		Detection," 2018 IEEE 5th International Conference on Data Science and
		Advanced Analytics (DSAA), 2018, pp. 209-216,
		doi: 10.1109/DSAA.2018.00031.
	"""

	def __init__(self, **kwargs):
		super(DRCN, self).__init__(**kwargs)
		self.encode = _Encoder(**kwargs)
		self.decode = _Decoder(**kwargs)
		self.classify = _Classifier(**kwargs)

	def call(self, inputs, training=None, **kwargs):
		if training:
			(source_pooled, target_pooled), convolved = self.encode(inputs)
			decoded = self.decode(target_pooled, convolved)
			classified = self.classify(source_pooled)
			result = (classified, decoded)
		else:
			encoded = self.encode(inputs)
			result = self.classify(encoded)
		return result

	def get_config(self):
		return super(DRCN, self).get_config()
