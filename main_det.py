"""
URL: https://towardsdatascience.com/implementing-real-time-object-detection-system-using-pytorch-and-opencv-70bac41148f7
"""
import torch
from torch import hub  # Hub contains other models like FasterRCNN
import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import time
from torchvision.models import detection
from obj_det_functions import load_saved_model ,score_frame,boxes, video_writer_config,yaml_loader,video_player
import socket, pickle, struct

# Creating a server socket
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = socket.gethostname()
print('HOST IP:',host_ip)
port = 9900
socket_address = (host_ip,port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at",socket_address)

# Using Saved model
model = load_saved_model()

# Checking for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load yaml file data
data = yaml_loader()  

# Obtaining the videoplayer-FD of the video to be played
video_player_fd = video_player()

# Video writer object
x_shape,y_shape,four_cc = video_writer_config(video_player_fd=video_player_fd)
# out = cv2.VideoWriter(data['Video']['save_path'],   # path
#                       four_cc,                      # codec type 
#                       20,                           # fps
#                       (x_shape, y_shape)          # width, height
#                       )  


while True:
    client_socket, addr = server_socket.accept()
    print('Got Connection from:', addr)
    if client_socket:
        # Reading frame-by-frame and scoring
        ret, frame = video_player_fd.read()  # Similar to do-while loop
        while ret:
            start_time = time()  # For measuring the FPS.

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = torch.from_numpy(frame).float()
            frame_actual = frame
            # blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)

            frame = torch.Tensor(frame)
            frame = torch.div(frame,255)  # change this

            frame = frame.permute(2,0,1).to(device=device)
            print('\nframe tensor = ',frame.shape)

            # Score the Frame
            results = score_frame(frame, model,device=device)  

            # Draw bounding boxes
            frame = boxes(frame=frame_actual,
                        results=results,
                        x_shape=x_shape,
                        y_shape=y_shape)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS.
            print(f"Frames Per Second : {fps}")

            a = pickle.dumps(frame)
            message = struct.pack("Q", len(a))+a
            client_socket.sendall(message)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_socket.close()
                video_player_fd.release()
                # out.release()
                cv2.destroyAllWindows()

            # out.write(np.uint8(frame))

            ret, frame = video_player_fd.read()  # Read frame for next iteration




