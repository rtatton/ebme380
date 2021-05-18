from tensorflow import keras
from tensorflow.keras import activations, layers, regularizers


def create_conv1d():
	return layers.Conv1D(
		filters=90,
		kernel_size=10,
		activation=activations.relu,
		kernel_regularizer=regularizers.l2,
		bias_regularizer=regularizers.l2,
		activity_regularizer=regularizers.l2)


class _Subencoder(layers.Layer):
	def __init__(self, **kwargs):
		super(_Subencoder, self).__init__(**kwargs)
		self.conv = create_conv1d()
		self.pool = layers.MaxPooling1D(pool_size=5)

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


class Encoder(layers.Layer):
	def __init__(self, **kwargs):
		super(Encoder, self).__init__(**kwargs)
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
		# TODO What is it?
		self.up_sample = layers.UpSampling1D(size=None)

	def call(self, inputs, training=None, **kwargs):
		if training:
			pooled_or_up_sampled, convolved = inputs
			convolved = self.conv(pooled_or_up_sampled)
			concatenated = self.cat(pooled_or_up_sampled, convolved)
			return self.up_sample(concatenated)


class Decoder(layers.Layer):
	def __init__(self, **kwargs):
		super(Decoder, self).__init__(**kwargs)
		self.decode1 = _Subdecoder(**kwargs)
		self.decode2 = _Subdecoder(**kwargs)
		self.conv = create_conv1d()

	def call(self, inputs, training=None, **kwargs):
		if training:
			pooled, (convolved1, convolved2) = inputs
			up_sampled = self.decode1((pooled, convolved1))
			up_sampled = self.decode2((up_sampled, convolved2))
			return self.conv(up_sampled)


class Classifier(layers.Layer):
	def __init__(self, **kwargs):
		super(Classifier, self).__init__(**kwargs)
		self.rnn = layers.Bidirectional(layers.GRU(
			units=80,
			kernel_regularizer=regularizers.l2,
			recurrent_regularizer=regularizers.l2,
			activity_regularizer=regularizers.l2))
		self.classify = layers.Activation(activations.sigmoid)

	def call(self, inputs, **kwargs):
		sequence = self.rnn(inputs)
		return self.classify(sequence)


class DRCN(keras.Model):

	def __init__(self, **kwargs):
		super(DRCN, self).__init__(**kwargs)
		self.encode = Encoder(**kwargs)
		self.decode = Decoder(**kwargs)
		self.classify = Classifier(**kwargs)

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
