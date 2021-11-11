"""
Functions for object detection
"""
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import cv2
import numpy as np
import yaml
import os

# Defining the classes
CLASSES = ["Car", "Truck", 'Bus', 'Person', 'Motorcycle']
# CLASSES = ["Car", "Truck"]

def yaml_loader(filepath = "./config.yaml"):
    """
    Loads the yaml file as file descriptor

    Args:
        filepath: Path of yaml file to be parsed
    Returns:
        data: 
    """
    with open (filepath, 'r') as file_descriptor:
        data = yaml.safe_load(file_descriptor)
    return data


def video_player():
    """
    Returns the FD of the video to be played 
    """
    data = yaml_loader()
    video_player = cv2.VideoCapture(data["Video"]["read_path"]) # Path of the video
    assert video_player.isOpened()  # Make sure that there is a stream
    
    return video_player


def video_writer_config(video_player_fd):
    """
    Configration for the video writer
    """
    # Video writer object to write our output stream.
    x_shape = int(video_player_fd.get(cv2.CAP_PROP_FRAME_WIDTH))   # property of frame (e.g. reads the width of the frame)
    y_shape = int(video_player_fd.get(cv2.CAP_PROP_FRAME_HEIGHT))  # property of frame (e.g. reads the width of the frame)
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # Using MJPEG codex

    return x_shape, y_shape, four_cc


def get_model(num_classes):
    """
    downloads the model and returns the model with the specified output classes
    Args:
        num_classes: number of output classes

    Returns:
        Returns the model with the linear layer of classes = num_classes
    """
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_saved_model():
    """ Loads model 
        Check config file the parameters
    """
    data = yaml_loader()
    model = get_model(data['Model']['num_classes'])  # Using Saved model
    model.load_state_dict(torch.load(data['Model']['path']))
    model.eval()  # put the model in evaluation mode
    return model


def parse_one_annot(path_to_data_file, filename):
    """

    Args:
        path_to_data_file: path to the csv file for parsing
        filename: name of the file to be

    Returns:

    """
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin",
                                                      "xmax", "ymax"]].values
    return boxes_array


def score_frame(frame, model,device):
    """
    The function below identifies the device which 
    is availabe to make the prediction and uses it 
    to load and infer the frame. 
    Once it has results it will extract the labels and 
    cordinates(Along with scores) for each object detected 
    in the frame.
    
    Args:
        frame: Image frame
        model: object detection model
    Returns:
        labels: label of the image
        cords: coordinates of the bounding boxes
    """
    # Model to device
    model.to(device)
    # Passing frame inside model
    results = model([frame])
    # Obtaining the output of the model and splitting it
    labels = results[0]["labels"].cpu().detach().numpy()
    cord = results[0]["boxes"].cpu().detach().numpy()
    scores = np.round(results[0]["scores"].cpu().detach().numpy(), decimals= 4)        
    # print('score =', scores)                
    # print('labels =', labels)
    # print('cords = ', cord,'\n')

    # labels = results.xyxyn[0][:, -1].cpu().numpy()
    # cord = results.xyxyn[0][:, :-1].cpu().numpy()

    return labels,scores, cord


def boxes(frame, results, x_shape, y_shape):
    """
    Provide bounding boxes for the frame

    Args:
        frame: video frame
        results: Output of the object detection model

    Returns:

    """
    # Boxes
    labels, scores, cord = results
    n = len(labels)
    for i in range(n):
        row = cord[i]
        # If score is less than 0.2 we avoid making a prediction.
        if scores[i] < 0.75:
            continue

        else: 
            # x1 = int(row[0] * x_shape)
            # y1 = int(row[1] * y_shape)
            # x2 = int(row[2] * x_shape)
            # y2 = int(row[3] * y_shape)
            # a = (x1*width,y*height) b = (x_max*width, y_max*height)

            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            # print(x1,'\t',y1,'\t',x2,'\t',y2)
            # print(row[0],'\t',row[1],'\t',row[2],'\t',row[3],'\n')

            bgr = (0, 255, 0)  # color of the box
            # classes = model.names  # Get the name of label index
            
            # print(CLASSES[0])
            obj_score=scores[i]
            class_name = "{}:{:.2f}%".format(CLASSES[int(labels[i]-1)],obj_score*100)
            print(class_name)
            # print("detected object score= ",obj_score)
            
            label_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the label.

            cv2.rectangle(img=frame,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=bgr,
                        thickness=3
                        )  # Plot the boxes
            cv2.putText(img=frame,
                        text=class_name, # text=classes[int(labels[i])]
                        org=(x1, y1),
                        fontFace=label_font,
                        fontScale=1,
                        color=bgr,
                        thickness=2
                        )  # Put a label over box.
    return frame
    