# Dataset downloaded from: http://data.csail.mit.edu/tofu/testset/vimeo_super_resolution_test.zip
# After that, run this code for LR and HR seperately to form a sorted data folder for convenience
# You might find helpful: how to detract the .DS_Store
# https://macpaw.com/how-to/remove-ds-store-files-on-mac
# find . -name '.DS_Store' -type f -delete
import os, sys, shutil

#source_folder = "Data/vimeo_super_resolution_test/low_resolution"
source_folder = "/Users/hanxiangren/Program/6.819/finalProject/FR-SRGAN/Data/vimeo_test_clean/sequences/"

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
    frames_list = os.listdir(video_path)
    frames_list.sort()
    for frames in frames_list:
        frames_path = os.path.join(video_path,frames)
        #print(frames_path)
        counter = counter + 1
        #new_frames_path = video_path + video + "_" + str(counter)
        new_frames_name = counter
        #print(new_frames_path)
        #os.rename(frames_path,new_frames_path)
        des = "Data/HR/" + str(new_frames_name)
        #des = "Data/LR/" + str(new_frames_name)
        print(des)
        shutil.copy(frames_path, des)

#Totally, we have 7824 frames-file for training
