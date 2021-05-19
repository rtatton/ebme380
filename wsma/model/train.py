import json
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import callbacks, losses, optimizers

from model import drcn

CHECKPOINT_DIR = "./ckpt"
if not os.path.exists(CHECKPOINT_DIR):
	os.makedirs(CHECKPOINT_DIR)


def get_compiled_model():
	model = drcn.DRCN()
	model.compile(
		optimizer=optimizers.Adam(),
		loss=[losses.BinaryCrossentropy(), losses.MeanSquaredError()])
	return model


def make_or_restore_model():
	checkpoints = [
		f'{CHECKPOINT_DIR}/{name}' for name in os.listdir(CHECKPOINT_DIR)]
	if checkpoints:
		latest_checkpoint = max(checkpoints, key=os.path.getctime)
		print("Restoring from", latest_checkpoint)
		return keras.models.load_model(latest_checkpoint)
	print("Creating a new model")
	return get_compiled_model()


def train():
	# TODO This is incomplete since the Distracted Driving dataset still
	#  needs to be preprocessed and integrated.
	model = make_or_restore_model()
	with open('mit-clean.txt', 'r') as f:
		data = np.array(json.load(f)['data'])
	signals, stress = np.expand_dims(data[:, :-1], 1), data[:, -1]
	model.fit(
		(signals, signals),
		(stress, signals),
		batch_size=32,
		callbacks=[
			callbacks.EarlyStopping(
				min_delta=1e-2,
				patience=2,
				verbose=1,
				restore_best_weights=True),
			callbacks.ModelCheckpoint(
				filepath='%s/ckpt-loss={loss:.2f}' % CHECKPOINT_DIR,
				save_freq=100,
				save_best_only=True,
				verbose=1)],
		validation_split=0.1,
		epochs=20)
