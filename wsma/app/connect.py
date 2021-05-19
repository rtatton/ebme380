import json

import serial


class WearableDevice:
	"""Data access object for the data streaming from the wearable device."""
	__slots__ = ('port', 'baud_rate', 'timeout', '_device')

	def __init__(
			self,
			port: str = 'COM7',
			baud_rate: int = 9600,
			timeout: float = 15.05):
		self.port = port
		self.baud_rate = baud_rate
		self.timeout = timeout
		self._device = serial.Serial(
			port=port, baudrate=baud_rate, timeout=timeout)

	def __call__(self):
		while True:
			if (line := self._device.readline()) is not None:
				yield json.loads(line)


if __name__ == '__main__':
	device = WearableDevice()
	for data in device():
		print(data)
