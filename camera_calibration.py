import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


def get_calibration_params(img_path, h_corners, v_corners, img_size, dst_file=None):
    objp = np.zeros((v_corners*h_corners, 3), np.float32)
    objp[:,:2] = np.mgrid[0:h_corners, 0:v_corners].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(img_path)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (h_corners, v_corners), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("Warning: failed to find corners for img %s" % fname)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    if dst_file:
        dist_pickle = {"mtx": mtx,
                       "dist": dist}
        with open(dst_file, 'wb') as outfile:
            pickle.dump(dist_pickle, outfile)

    return mtx, dist

if __name__ == "__main__":
    test_image = './camera_cal/calibration1.jpg'
    img = cv2.imread(test_image)
    img_size = (img.shape[1], img.shape[0])

    mtx, dist = get_calibration_params('./camera_cal/calibration*.jpg', 9, 6,
                                       img_size, dst_file='./camera_cal/calibration.p')
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show(block=True)
