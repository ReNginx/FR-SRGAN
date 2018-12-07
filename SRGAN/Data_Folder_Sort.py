# Dataset downloaded from: http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip
# After that, run this code for LR and HR seperately to form a sorted data folder for convenience
# You might find helpful: how to detract the .DS_Store
# https://macpaw.com/how-to/remove-ds-store-files-on-mac
# find . -name '.DS_Store' -type f -delete
import os, sys, shutil

#source_folder = "Data/vimeo_super_resolution_test/low_resolution"
source_folder = "Data/LR"

if os.path.exists(source_folder + ".DS_Store"):
    os.remove(source_folder + ".DS_Store")
else:
    print(".DS_Store does not exist")

video_list = os.listdir(source_folder)
video_list.sort()
#print(video_list)

counter = 0

for video in video_list:
    video_path = os.path.join(source_folder, video)
    frames = os.listdir(video_path)
    frames.sort()
    for frame in frames:
        frame_path = os.path.join(video_path,frame)
        #print(frame_path)
        counter = counter + 1
        #print(counter)
        #new_frames_path = video_path + video + "_" + str(counter)
        new_frames_name = counter
        #print(new_frames_path)
        #os.rename(frames_path,new_frames_path)
        des = "Data/LR_new/" + str(counter) + '.jpg'
        #des = "Data/LR/" + str(new_frames_name)
        print(des)
        shutil.copy(frame_path, des)

#Totally, we have 7824 frames-file for training
