import cv2
import os

def save_videos_frame_from_path_file(video_path = "./VideosPaths", output_path = "./VideosSampleFrames", freq = 25):
    '''
    Save video frames from a list of video paths in a text file
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Storing frames path:" + output_path)
                          
    with open(video_path, 'r') as f:
        paths = f.read().split("\n")
    #     print(len(content))
#         print(paths)
        for v_path in paths:
            print(v_path)
            video_name = v_path.split("/")[-1]
            video_frame_path = "{0}/{1}".format(output_path, video_name)
            if not os.path.exists(video_frame_path):
                os.makedirs(video_frame_path)
                print("Storing frames path:" + video_frame_path)
                vcap = cv2.VideoCapture(v_path)
                if not vcap.isOpened():
                    exit(0)

                #Capture images per 25 frame
                frameFrequency = freq

                #iterate all frames
                total_frame = 0
                id = 0
                while True:
                    ret, frame = vcap.read()
                    if ret is False:
                        break
                    total_frame += 1
                    if total_frame%frameFrequency == 0:
                        id += 1
                        cv2.imwrite("{0}/{1}_{2}.jpeg".format(video_frame_path, id, video_name), frame)
                        
                vcap.release()

save_videos_frame_from_path_file(video_path = "video_paths.txt", output_path = "./cockpitview" , freq = 30)