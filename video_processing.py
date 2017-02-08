import cv2
import argparse
import pickle
import numpy as np
import image_processing as ip
from moviepy.editor import VideoFileClip


def process_frame(img):
    global mtx
    global dist
    global left_marker
    global right_marker

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
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    src_pts = [[420, 570], [265, 680], [1050, 680], [876, 570]]
    dst_pts = [[200, 570], [200, 680], [1080, 680], [1080, 570]]
    warped = ip.perspective_transform(combined, src_pts, dst_pts)

    # todo: store fit, reuse fit, measure fit certainty
    left_marker, right_marker = ip.poly_fit(warped, left_marker, right_marker)

    # Re-build frame
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    warped_lane = ip.annotate_poly_fit(color_warp, left_marker.poly_fit_px(), right_marker.poly_fit_px())

    annotated_lane = ip.perspective_transform(warped_lane, dst_pts, src_pts)
    result = cv2.addWeighted(img, 1, annotated_lane, 0.3, 0)

    radius = (left_marker.curvature_radius() + right_marker.curvature_radius()) / 2
    radius_str = "Curvature radius: {0:.1f}m".format(radius)
    cv2.putText(result, radius_str, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

    return result


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

    left_marker = None
    right_marker = None

    in_video = VideoFileClip(args.input)
    out_video = in_video.fl_image(process_frame)
    out_video.write_videofile(args.output, audio=False)

