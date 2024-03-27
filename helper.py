import cv2
import os
import shutil
output_file = 'temp.mp4'

def del_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)

def frame_to_vid(frames_dir):
    if os.path.isfile(output_file):
        os.remove(output_file)
    
    d = {int(i.split('_')[-1].split('.')[0]):i for i in os.listdir(frames_dir)}
    d = {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}
    
    frame = cv2.imread(os.path.join(frames_dir, os.listdir(frames_dir)[0]))
    frame_height, frame_width, channels = frame.shape
    fps = 30

    codec = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, codec, fps, (frame_width, frame_height))
    # Loop through the frames in the directory and write each frame to the output video file
    for filename in sorted(list(d.values())):
        frame = cv2.imread(os.path.join(frames_dir, filename))
        out.write(frame)

    # Release the video writer object
    out.release()