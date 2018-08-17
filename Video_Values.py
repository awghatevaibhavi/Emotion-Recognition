'''

This programs displays videos with arousal and valence displayed on it.

'''

import cv2
import csv

input_file = []
with open('Data/labels/Test_02.csv') as csvfile:
    attributes = csv.reader(csvfile)
    for element in attributes:
        input_file.append(';'.join(element))

        # converts it into 2D array
for index in range( len(input_file)):
    input_file[index] = input_file[index].split(';')

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('Data/video/Test_02.avi')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

index = 0

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True and index < 1740:
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if (pos_frame - 1) % 5 == 0:
            # Display the resulting frame
            cv2.putText(frame, 'Arousal: ' + input_file[index][0], (50,50), cv2.FONT_ITALIC, 0.8, (0,0,255),)
            cv2.putText(frame, 'Valence: ' + input_file[index][1], (50, 100), cv2.FONT_ITALIC, 0.8, (255, 0, 0))
            cv2.imshow('Frame', frame)
            index = index + 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()