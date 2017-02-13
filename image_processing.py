import numpy as np
import cv2
from skimage import exposure


def rescale_intensity(img, perc_a, perc_b):
    pa, pb = np.percentile(img, (perc_a, perc_b))
    img_rescale = exposure.rescale_intensity(img, in_range=(pa, pb))
    return img_rescale


def adjust_brightness(img, factor):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    y, u, v = cv2.split(yuv)
    y = np.asarray(y, dtype=np.int32)
    y += factor
    y = np.asarray(np.clip(y, 0, 255), dtype=np.uint8)
    yuv = cv2.merge((y, u, v))
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    x = True if orient == 'x' else False
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1 if x else 0, 1 if not x else 0, ksize=sobel_kernel)
    sobel = np.absolute(sobel)
    sobel = np.uint8(255 * sobel / np.max(sobel))
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel >= thresh[0]) & (sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(mag_sobelxy)/255
    sobel = (mag_sobelxy/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel >= mag_thresh[0]) & (sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    gradient_dir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(gradient_dir)
    binary_output[(gradient_dir >= thresh[0]) & (gradient_dir <= thresh[1])] = 1
    return binary_output


def perspective_transform(img, src_pts, dst_pts):
    img_size = (img.shape[1], img.shape[0])
    m = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    return warped


def get_yellow_white_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    w_lower = np.array([0, 0, 200])
    w_upper = np.array([180, 255, 255])
    mask_w = cv2.inRange(hsv, w_lower, w_upper)
    y_lower = np.array([20, 100, 100])
    y_upper = np.array([30, 255, 255])
    mask_y = cv2.inRange(hsv, y_lower, y_upper)
    return mask_y | mask_w


def apply_yellow_white_mask(image):
    masked = cv2.bitwise_and(image, image, mask=get_yellow_white_mask(image))
    return masked
