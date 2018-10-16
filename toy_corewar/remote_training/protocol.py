import json
import socket

class Protocol:
	MAGIC = b'GOLAI###'

	def __init__(self, socket):
		self.socket = socket

	def send_obj(self, obj):
		msg = json.dumps(obj).encode('utf-8')
		self.socket.send(self.MAGIC + str(len(msg)).rjust(16, ' ').encode("ascii"))
		self.socket.send(msg)

	def recv_obj(self):
		header = self.socket.recv(24)
		if header[:8] != self.MAGIC:
			return #handle error
		length = int(header[8:].decode('utf-8').strip())
		msg = self.socket.recv(length)
		return json.loads(msg)
