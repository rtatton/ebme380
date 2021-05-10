import json

import attr
import serial


@attr.s(slots=True)
class Arduino:
	port = attr.ib(type=str, default='COM7', kw_only=True)
	baud_rate = attr.ib(type=int, default=9600, kw_only=True)
	timeout = attr.ib(type=float, default=15.05, kw_only=True)
	_device = attr.ib(type=serial.Serial, init=False, repr=False)

	def __attrs_post_init__(self):
		self._device = serial.Serial(
			port=self.port,
			baudrate=self.baud_rate,
			timeout=self.timeout)

	def stream(self):
		while True:
			yield json.loads(self._device.readline())


if __name__ == '__main__':
	device = Arduino()
	for data in device.stream():
		print(data)
