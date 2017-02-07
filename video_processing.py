import cv2
import argparse
import pickle
import numpy as np
import image_processing as ip
from moviepy.editor import VideoFileClip


def process_frame(img):
    global mtx
    global dist

    img = cv2.undistort(img, mtx, dist, None, mtx)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = hsv[:, :, 1]
    
    ksize = 5  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = ip.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(50, 150))
    grady = ip.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(50, 150))
    mag_binary = ip.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(50, 150))
    dir_binary = ip.dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 255

    src_pts = [[420, 570], [265, 680], [1050, 680], [876, 570]]
    dst_pts = [[200, 570], [200, 680], [1080, 680], [1080, 570]]
    warped = ip.perspective_transform(combined, src_pts, dst_pts)

    return cv2.merge((warped, warped, warped))
    #return warped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driving Lane Detection')
    parser.add_argument('--input', default='./short_project_video.mp4', help='Path to input video file.')
    parser.add_argument('--output', default='./output_videos/short_project_video.mp4', help='Path to output video file.')
    parser.add_argument('--calibration_file', default='./camera_cal/calibration.p', help='Camera calibration params')
    args = parser.parse_args()

    with open(args.calibration_file, 'rb') as pfile:
        cal_dict = pickle.load(pfile)
        mtx = cal_dict["mtx"]
        dist = cal_dict["dist"]

    in_video = VideoFileClip(args.input)
    out_video = in_video.fl_image(process_frame)
    out_video.write_videofile(args.output, audio=False)

