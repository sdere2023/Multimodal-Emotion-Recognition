import cv2
import os

# define the path to your video files
video_folder_path = 'Final Project/Video Model/Videos/'

# loop through each file in the folder
for filename in os.listdir(video_folder_path):
    # check if the file is a video file
    if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
        # get the full path to the file
        video_file_path = os.path.join(video_folder_path, filename)
        # open the video file
        cap = cv2.VideoCapture(video_file_path)
        # get the frame rate of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # loop through each frame of the video
        while cap.isOpened():
            # read the frame
            ret, frame = cap.read()
            # check if the frame was read successfully
            if ret:
                # define the path to save the frame
                frame_file_path = os.path.join(video_folder_path, filename[:-4], str(cap.get(cv2.CAP_PROP_POS_FRAMES)).zfill(6) + '.jpg')
                # create the directory if it does not exist
                os.makedirs(os.path.dirname(frame_file_path), exist_ok=True)
                # save the frame as a JPEG image
                cv2.imwrite(frame_file_path, frame)
            else:
                break
        # release the video capture object
        cap.release()
