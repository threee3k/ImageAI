import numpy as np 
import cv2 

#Setting up some paths
image_path = ''
prototext_path = 'model/MobileNetSSD_deploy.prototxt'
model_path = 'models/MobileNetSSD_deploy.caffemodel'

# To avoid unneccesary errors

min_confidence = 0.2

# Stuff that the script detects

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtabel", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Generate the rectangles around recognised objects

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Creating the neural network aka object detection
net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)

# Number of your camera
cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()

    height, width = image.shape[0], image.shape[1]

    blob = cv2.dnn .blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)

    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])
            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            predection_text = f"{classesclasses[class_index]}: {confidence:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(image, predection_text, (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y +15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)


    cv2.imshow("Detected Objects", image)
    # 5 is the FPS of the camera
    cv2.waitKet(5)
cv2.destroyAllWindows()
cap.realease()




