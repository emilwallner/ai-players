import os
import sys
import socket
import json

from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, \
	QMessageBox, QTreeView
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPainter, QPixmap

from .GUI.ui.ui_mainwindow import Ui_MainWindow

from .protocol import Protocol

class MainWindow(QMainWindow):
	def __init__(self):
		super(QMainWindow, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)
		self.ui.btnConnect.setText("Connect")
		self.ui.textEdit.setText("hello")

		self.sock = None

		self.ui.btnConnect.clicked.connect(self.connect)
		path = os.path.dirname(os.path.realpath(__file__))

		pic = QPixmap(os.path.join(path, "GUI/res/dxe3z8g39ks11.png"))
		self.ui.tempImg.setPixmap(pic)

		self.ui.tabWidget.setCurrentIndex(0)

	def connect(self):
		try:
			host = str(self.ui.connHostname.displayText())
			port = self.ui.connHostport.value()
			self.sock = socket.socket()
			self.sock.connect((host, port))
			self.ui.btnConnect.setText("Disconnect")
			self.ui.tabWidget.setCurrentIndex(1)
			self.get_config()
		except Exception as e:
			self.ui.statusbar.showMessage(str(e), 4000)

	def get_config(self):
		try:
			p = Protocol(self.sock)
			p.send_obj({"command": "config"})
			resp = p.recv_obj()
			print(resp)
			self.config = resp
			self.ui.textEdit.setPlainText(json.dumps(self.config, indent = 4))
			self.ui.textEdit.repaint	()
		except Exception as e:
			self.ui.statusbar.showMessage(str(e), 4000)

app = QApplication(sys.argv)
window = MainWindow()
window.show()

app.exec_()
