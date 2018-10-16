import config
config.load("config.json")
import socket
import json
import multiprocessing
#add ssl later

trainingcfg = config.get_cfg()

from protocol import Protocol

def get_config(sock, data):
	p = Protocol(sock)
	p.send()

def ping(sock, data):
	p = Protocol(sock)
	p.send_obj({"message": "alive"})

commands = {
	'config': get_config,
	'ping':	ping,
	'exit': None
}

MAGIC = b'GOLAI###'

def handler(conn):
	while True:
		p = Protocol(conn)
		data = p.recv_obj()
		try:
			if not data or not "command" in data.keys():
				print("No command received. closing connection")
				break
			if data["command"] in commands.keys():
				if data["command"] == "exit":
					conn.shutdown(socket.SHUT_RD)
					break
				commands[data["command"]](conn, data)
			else:
				p.send_obj({"message": "command {} doesn't exists".format(data["command"])})
		except Exception as e:
			print(e)
			p.send_obj({"message": str(e)})


sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", 8888))

sock.listen()

while True:
	try:
		conn, addr = sock.accept()
		print("connection received from: {}".format(addr))
		handler(conn)
		conn.close()
		print("connection closed")
	except KeyboardInterrupt:
		print("exiting server...")
		if 'conn' in locals():
			conn.close()
		sock.close()
		break
