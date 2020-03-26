import cv2 as cv
import numpy as np
from flask import Flask, request, jsonify
from os import listdir

#Write down conf, nms thresholds,inp width/height
confThreshold = 0.25
nmsThreshold = 0.40
inpWidth = 416
inpHeight = 416

#Load names of classes and turn that into a list
classesFile = "coco.names"
classes = None

#Model configuration
modelConf = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def postprocess(frame, outs, framename, video_content):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confThreshold:
                centerX = int(detection[0] * frame_width)
                centerY = int(detection[1] * frame_height)

                width = int(detection[2]*frame_width)
                height = int(detection[3]*frame_height)

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        video_content = append_pred_to_vehi(framename, video_content, classIDs[i], confidences[i], left, top, left + width, top + height)
    return video_content

def append_pred_to_vehi(framename, video_content, classId, conf, left, top, right, bottom):
    # Confidence pourcentage
    label = '%.2f' % conf

    # Get the label for the class name and its confidence    
    if classes:
        assert (classId < len(classes))
        vehicle_type = classes[classId]

        if vehicle_type in ["car", "truck"]:
            # Add new box in the result variable
            new_box = {
                "object": vehicle_type,
                "proba": label,
                "left": left,
                "bot": bottom,
                "right": right,
                "top": top,
            }

        try:
            video_content[framename].append(new_box)
        except:
            video_content[framename] = []
    return video_content

def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()    

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


############ MAIN ##############
app = Flask(__name__)

@app.route('/')
def main_page():
    video_ref = request.args.get("v")
    if video_ref is None:
        return f'You have to specify the video ID'

    video_content = {}
    try:
        frames = listdir("videos/" + video_ref)
    except FileNotFoundError:
        return f'No video is saved with this ID'

    for framename in frames:
        frame = cv.imread("videos/" + video_ref + "/" + framename)

        # Create a 4D blob from a frame
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Set the videos the the net
        net.setInput(blob)
        outs = net.forward(get_outputs_names(net))
        video_content = postprocess(frame, outs, framename, video_content)

    return jsonify(video_content)

if __name__ == "__main__":
    # Set up the net
    net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    app.run()
















