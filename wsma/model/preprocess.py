import json
import os
import re
import shutil

import numpy as np
import pandas as pd
from scipy import signal


# noinspection LongLine
def preprocess_mit_dataset(src: str, dest: str):
	"""Preprocess the MIT Driver Stress dataset.

	Args:
		src (str): Filepath of the original dataset.
		dest (str): Filepath of the preprocessed dataset.

	Returns:
		None.

	References:
		https://github.com/chriotte
		/Stress_classifier_with_AutoML_and_wearable_devices
		https://dev.to/kriyeng/8-tips-for-google-colab-notebooks-to-take
		-advantage-of-their-free-of-charge-12gb-ram-gpu-be4
	"""
	df = pd.read_csv(src).reset_index(drop=True)
	df = df[['HR', 'handGSR', 'stress']]
	df.rename(columns={'handGSR': 'edr', 'HR': 'hr'}, inplace=True)
	df['stress'] = np.where(df['stress'] >= 0.5, 1, 0)
	df.replace((np.inf, -np.inf), np.nan, inplace=True)
	df[~np.isfinite(df)] = np.nan
	df.fillna(df.mean(), inplace=True)
	df['hr'] = signal.medfilt(df['hr'], kernel_size=13)
	data = {'fields': ['hr', 'edr', 'stress'], 'data': df.to_numpy().tolist()}
	json.dump(data, open(dest, 'w'))


def preprocess_distracted_dataset(src: str, dest: str, zipped: bool = True):
	"""Preprocesses the Distracted Driving "Structured Study Data" dataset.

	Args:
		src (str): Filepath of the original dataset.
		dest (str): Filepath of the preprocessed dataset.
		zipped (bool): True indicates the "Structured Study Data" folder is
			zipped.

	Returns:
		None.

	References:
		https://doi.org/10.17605/OSF.IO/C42CN
	"""
	reg = re.compile(r'.(HR|peda|tp|bar)')
	if not os.path.exists(dest):
		os.mkdir(dest)
	if zipped:
		shutil.unpack_archive(src, dest)
		for f in os.listdir(dest):
			shutil.unpack_archive(f, dest)
	for root, dirs, files in os.walk(dest if zipped else src):
		for file in filter(lambda x: re.search(reg, x), files):
			shutil.copy(os.path.join(root, file), os.path.join(dest, file))


if __name__ == '__main__':
	src = input('src:')
	dest = input('dest:')
	zipped = bool(input('zipped:'))
	preprocess_distracted_dataset(src, dest, zipped=zipped)
