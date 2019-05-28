import base64
import cv2
import zmq
import socket
import threading
import time

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://143.248.253.61:5555')

camera = cv2.VideoCapture(0)

server_address = ""
port = 8040
size = 1024
ev3M = 'stop'

def com_client_thread(client, clientInfo):
    global ev3M
    while True:
        data = client.recv(size).decode()
        if data:
            print(data)
            ev3M = data
        else:
            print('Computer Disconnected')
            client.close()
            break

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((server_address, port))
sock.listen(2)
#sock.listen(1)

print("Waiting for Clients...")
com_client, com_clientInfo = sock.accept()

t1 = threading.Thread(target = com_client_thread, args = (com_client, com_clientInfo))
t1.start()
print("Connected Computer client:", com_clientInfo)

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 12, (640,480))

def frame_thread(camera, out):
    while True:
        grabbed, frame = camera.read()
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)
        out.write(frame)

t2 = threading.Thread(target = frame_thread, args = (camera, out))
t2.start()
print('frame thread started')

client, clientInfo = sock.accept()
print("Connected EV3 client:", clientInfo)

while True:
    data = client.recv(size).decode()
    if data:
        #print(data)
        client.send(ev3M.encode())
        ev3M = 'stop'
    else:
        print("EV3 disconnected")
        client.close()
        break
sock.close()
