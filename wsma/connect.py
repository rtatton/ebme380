import json

import serial


class Arduino:
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

	def stream(self):
		while True:
			yield json.loads(self._device.readline())


if __name__ == '__main__':
	device = Arduino()
	for data in device.stream():
		print(data)
