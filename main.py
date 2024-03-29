import os
from app import app
from flask import flash, request, redirect, url_for, render_template, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np


@app.route('/')
def upload_form():
    return render_template('upload.html')

#method to upload video
@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('no file selected')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return Response(get_model(os.path.join(app.config['UPLOAD_FOLDER'], filename)), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/display/<filename>')
def display_video(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

#method to detect objects
def get_model(file_name):
    while True:
        # get the video file as stream using cv2
        video_stream = cv2.VideoCapture(file_name)

        # create a while loop till the stream is open
        while (video_stream.isOpened):
            # get the current frame from video stream
            ret, current_frame = video_stream.read()
            # use the video current frame instead of image
            img_to_detect = current_frame
            img_height = img_to_detect.shape[0]
            img_width = img_to_detect.shape[1]
            # resize to match input size, convert to blob to pass into model
            resized_img_to_detect = cv2.resize(img_to_detect, (300, 300))
            # recommended scale factor 0.007843, height 300, width 300 and mean of 255 is 127.5
            img_blob = cv2.dnn.blobFromImage(resized_img_to_detect, 0.005, (300, 300), 127.5)

            # set of 21 class labels in alphabetical order (background + rest of 20 classes)
            class_labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"]
            # Loading pretrained model from prototext and caffemodel files
            # input preprocessed blob into model and pass through the model
            # get the detection predictions using forward()
            mobilenetssd = cv2.dnn.readNetFromCaffe('Datasets/mobilenetssd.prototext',
                                                    'Datasets/mobilenetssd.caffemodel')
            mobilenetssd.setInput(img_blob)
            obj_detections = mobilenetssd.forward()
            no_of_detections = obj_detections.shape[2]
            print("No. of detection: ", no_of_detections)
            obj_count_dict = {}
            # loop over the detections
            for index in np.arange(0, no_of_detections):
                prediction_confidence = obj_detections[0, 0, index, 2]
                # take only predictions with confidence more than 50%
                if prediction_confidence > 0.50:

                    # get the predicted label
                    predicted_class_index = int(obj_detections[0, 0, index, 1])
                    predicted_class_label = class_labels[predicted_class_index]

                    # get the each label count
                    if predicted_class_label in obj_count_dict:
                        v = obj_count_dict[predicted_class_label]
                        obj_count_dict[predicted_class_label] = v + 1
                    else:
                        obj_count_dict[predicted_class_label] = 1

                    # obtain the bounding box co-oridnates for actual image from resized image size
                    bounding_box = obj_detections[0, 0, index, 3:7] * np.array(
                        [img_width, img_height, img_width, img_height])
                    (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")

                    # print the prediction in console
                    predicted_label = "{}: {:.2f}% :#:{}".format(class_labels[predicted_class_index],
                                                                 prediction_confidence * 100,
                                                                 obj_count_dict[predicted_class_label])
                    print("predicted object {}: {}".format(index + 1, predicted_label))

                    # draw rectangle and text in the image
                    cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0, 255, 0), 2)
                    cv2.putText(img_to_detect, predicted_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

            ret, buffer = cv2.imencode('.jpg', img_to_detect)
            img_to_detect = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_to_detect + b'\r\n')


if __name__ == "__main__":
    app.run()