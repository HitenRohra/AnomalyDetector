import cv2
import numpy as np
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")

SEQUENCE_LENGTH = 20
IMAGE_HEIGHT, IMAGE_WIDTH = (64,64)
CLASSES_LIST = ['Robbery', 'Explosion', 'Shoplifting', 'Arrest', 'Fighting', 'RoadAccidents']
model_path = os.path.join('model','model.h5')
convlstm_model = tf.keras.models.load_model(model_path, compile=False)


def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

def vid_class_pred(path,class_list):
    arr = np.array(frames_extraction(path))
    arr = np.expand_dims(arr, axis=0)
    model_pred = convlstm_model.predict(arr).ravel()
    pred_prob = max(model_pred)
    pred_class = class_list[np.argmax(model_pred)]
    return pred_class,pred_prob

def open_vid(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): 
        print("Error opening video stream or file")
    return cap

def add_box(img, h, w):
    min_dim = min(h, w)
    box_size = int(min_dim / 1.5)
    x1 = np.random.randint(0, w - box_size)
    y1 = np.random.randint(0, h - box_size)
    x2 = x1 + box_size
    y2 = y1 + box_size
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
    return x1, y1, x2, y2

def main():
    vid_path = os.path.join('sample','uploaded_video.mp4')
    output_video_path = "./sample/output_video.mp4"
    cap = open_vid(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    anomaly_flag = False
    box_coords = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w, _ = frame.shape
            if not anomaly_flag:
                pred_class, pred_prob = vid_class_pred(vid_path,CLASSES_LIST)
                if pred_class != 'Normal':
                    if box_coords is None:
                        x1, y1, x2, y2 = add_box(frame, h, w)
                        box_coords = (x1, y1, x2, y2)
                    else:
                        x1, y1, x2, y2 = box_coords
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                        cv2.putText(frame, str(pred_prob), (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 1)
                    anomaly_flag = True
            else:
                x1, y1, x2, y2 = box_coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, str("{:.2f}".format(pred_prob)), (x1+5, y1+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)

            out.write(frame)
            # cv2.imshow('Frame',frame)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        else: 
            break
        

    # cap.release()
    # cv2.destroyAllWindows()
    
    return anomaly_flag
