# mp4 get all images frame by frame python
import cv2


def get_images(file):
    video_cap = cv2.VideoCapture(file)
    success, image = video_cap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = video_cap.read()
        print('Read a new frame: ', success)
        count += 1


if __name__ == "__main__":
    file_source = 'newtest720.mp4'
    get_images(file_source)
