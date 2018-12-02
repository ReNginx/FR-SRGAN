# Dataset downloaded from: http://data.csail.mit.edu/tofu/testset/vimeo_test_clean.zip
# https://macpaw.com/how-to/remove-ds-store-files-on-mac
import os, sys, shutil

source_folder = "Data/vimeo_super_resolution_test/low_resolution"
video_list = os.listdir(source_folder)

#print(video_list)

for video in video_list:
    counter = 0
    video_path = os.path.join(source_folder, video)
    frames_list = os.listdir(video_path)
    for frames in frames_list:
        frames_path = os.path.join(video_path,frames)
        print(frames_path)
        counter = counter + 1
        #new_frames_path = video_path + video + "_" + str(counter)
        new_frames = video + "_" + str(counter)
        #print(new_frames_path)
        #os.rename(frames_path,new_frames_path)
        des = "Data/LR/" + new_frames
        shutil.move(frames_path, des)



