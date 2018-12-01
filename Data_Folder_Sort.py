# Dataset downloaded from: http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip
import os, sys, shutil

source_folder = "Data/"

video_list = os.listdir(source_folder)

for video in video_list:
    counter = 0
    video_path = os.path.join(source_folder, video)
    frames_list = os.listdir(video_path)
    for frames in frames_list:
        frames_path = os.path.join(video_path,frames)
        counter = counter + 1
        os.rename(frames,video + "_" + counter)
        des = "Data/LR/" + frames
        shutil.move(frames_path, des)
        


