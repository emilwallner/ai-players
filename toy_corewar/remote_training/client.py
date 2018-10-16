import socket
import json

from protocol import Protocol

sock = socket.socket()
sock.connect(("localhost", 8888))

while True:
	try:
		command = input("$> ")
		p = Protocol(sock)
		p.send_obj({"command": command})
		resp = p.recv_obj()
		if not resp:
			break
		print(resp['message'	])
	except KeyboardInterrupt:
		break

sock.close()
