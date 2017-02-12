import cv2
import numpy as np
import image_processing as ip
from LaneMarker import LaneMarker


class LaneFinder:

    def __init__(self,
                 kernel_size,
                 cam_mtx,
                 cam_dist,
                 s_thr=(120, 255),
                 l_thr=50,
                 sobel_thr=(20, 255),
                 window_width=100,
                 num_windows=9,
                 pixel_thr=50,
                 debug = False):
        self._kernel_size = kernel_size
        self._cam_mtx = cam_mtx
        self._cam_dist = cam_dist
        self._s_thr = s_thr
        self._l_thr = l_thr
        self._sobel_thr = sobel_thr
        self._window_width = window_width
        self._num_windows = num_windows
        self._pixel_thr = pixel_thr
        self._debug = debug
        self._debug_log = []

        self._previous_offset = []
        self._previous_l = []
        self._previous_r = []
        self._failure_cnt = 0
        self._lane_departure_thr = 0.5

    def process_frame(self, img):
        img_width = img.shape[1]
        img_height = img.shape[0]

        # Perspective transform coordinates
        src_pts = [[img_width * 0.439, img_height * 0.65],
                   [img_width * 0.561, img_height * 0.65],
                   [img_width * 0.76, img_height * 0.935],
                   [img_width * 0.24, img_height * 0.935]]
        dst_pts = [[img_width * 0.25, 0],
                   [img_width * 0.75, 0],
                   [img_width * 0.75, img_height],
                   [img_width * 0.25, img_height]]

        # Undistort image
        undist = cv2.undistort(img, self._cam_mtx, self._cam_dist, None, self._cam_mtx)

        # HLS part of the pipeline
        hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
        s = hls[:, :, 2]
        l = hls[:, :, 1]
        s_binary = np.zeros_like(s)
        s_binary[(s > self._s_thr[0]) & (s <= self._s_thr[1]) & (l > self._l_thr)] = 1

        # RGB part of the pipeline
        r = undist[:, :, 0]
        x_gradient = ip.abs_sobel_thresh(r, orient='x', sobel_kernel=self._kernel_size, thresh=self._sobel_thr)
        y_gradient = ip.abs_sobel_thresh(r, orient='y', sobel_kernel=self._kernel_size, thresh=self._sobel_thr)

        combined = np.zeros_like(x_gradient)
        combined[((x_gradient == 1) & (y_gradient == 1)) |
                 (s_binary == 1)] = 1

        warped = ip.perspective_transform(combined, src_pts, dst_pts)

        if len(self._previous_l) and len(self._previous_r) and self._failure_cnt < 3:
            left_marker, right_marker, warped_dbg = self.find_lanes_poly_fit(warped,
                                                                             self._previous_l[-1],
                                                                             self._previous_r[-1])
            left_marker, right_marker = self.verify_markers(left_marker, right_marker)
        else:
            left_marker, right_marker, warped_dbg = self.find_lanes_histogram(warped)
            self._failure_cnt = 0
            left_marker, right_marker = self.verify_markers(left_marker, right_marker)

        self._previous_l.append(left_marker)
        self._previous_r.append(right_marker)

        roc_l = np.mean([m.curvature_radius() for m in self._previous_l[-15:]])
        roc_r = np.mean([m.curvature_radius() for m in self._previous_r[-15:]])
        offset = self.car_offset(warped.shape, left_marker, right_marker, 0.25 / 0.24)
        self._previous_offset.append(offset)

        # Re-build frame
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        warped_lane = self.annotate_lane(color_warp, left_marker, right_marker)

        annotated_lane = ip.perspective_transform(warped_lane, dst_pts, src_pts)
        result = cv2.addWeighted(img, 1, annotated_lane, 0.3, 0)
        radius_str = "Curvature L: {0:5.0f}m R: {1:5.0f}m".format(roc_l, roc_r)
        cv2.putText(result, radius_str, (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))
        offset_str = "Car offset: {0:1.2f}m".format(offset)
        cv2.putText(result, offset_str, (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255))

        if self._debug:
            self._debug_log.append({'left': left_marker,
                                    'right': right_marker,
                                    'result': result,
                                    'undistorted': undist,
                                    's': s,
                                    'l': l,
                                    'red': r,
                                    's_binary': s_binary,
                                    'x_gradient': x_gradient,
                                    'y_gradient': y_gradient,
                                    'combined': combined,
                                    'warped': warped_dbg,
                                    'offset': offset,
                                    'failed': self._failure_cnt})

        return result

    def verify_markers(self, left_marker, right_marker):
        failed = False
        if not left_marker:
            left_marker = self._previous_l[-1]
            failed = True
        if not right_marker:
            right_marker = self._previous_r[-1]
            failed = True

        if failed:
            self._failure_cnt += 1
        else:
            self._failure_cnt = 0
        return left_marker, right_marker

    def debug_log(self):
        return self._debug_log

    def car_offset(self, shape, left_marker, right_marker, warp_factor):        
        y = shape[0] - 1
        l_fit = left_marker.poly_fit_px()
        lx = l_fit[0] * y ** 2 + l_fit[1] * y + l_fit[2]
        r_fit = right_marker.poly_fit_px()
        rx = r_fit[0] * y ** 2 + r_fit[1] * y + r_fit[2]
        center = shape[1] / 2
        px_offset = (center - ((rx + lx) / 2)) * warp_factor
        m_offset = px_offset * LaneMarker.xm_per_px
        return m_offset

    def annotate_lane(self, img, left_marker, right_marker):
        """
        Draws the lane over an existing image
        """
        left_fit = left_marker.poly_fit_px()
        right_fit = right_marker.poly_fit_px()

        plot = np.linspace(0, img.shape[0] - 1, img.shape[0])
        lx = left_fit[0] * plot ** 2 + left_fit[1] * plot + left_fit[2]
        rx = right_fit[0] * plot ** 2 + right_fit[1] * plot + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([lx, plot]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rx, plot])))])
        pts = np.hstack((pts_left, pts_right))

        color = (0, 255, 0) if abs(self._previous_offset[-1]) <= self._lane_departure_thr else (255, 0, 0)
        cv2.fillPoly(img, np.int_([pts]), color)
        return img

    def find_lanes_poly_fit(self, img, left_marker, right_marker):
        """
        Finds lane pixels using sliding windows.
        Search is centered around an existing polynomial fit
        """
        # Debug image
        dbg_img = None
        if self._debug:
            dbg_img = np.dstack((img, img, img)) * 255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        l_fit = left_marker.poly_fit_px()
        r_fit = right_marker.poly_fit_px()

        left_lane_ind = []
        right_lane_ind = []

        window_height = np.int(img.shape[0] / self._num_windows)
        for window in range(self._num_windows):
            # Defining window boundaries for left and right lanes
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height

            lw_ind = ((nonzero_x > (l_fit[0] * (nonzero_y ** 2) + l_fit[1] * nonzero_y + l_fit[2] - self._window_width)) &
                      (nonzero_x < (l_fit[0] * (nonzero_y ** 2) + l_fit[1] * nonzero_y + l_fit[2] + self._window_width)) &
                      ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high))).nonzero()[0]
            rw_ind = ((nonzero_x > (r_fit[0] * (nonzero_y ** 2) + r_fit[1] * nonzero_y + r_fit[2] - self._window_width)) &
                      (nonzero_x < (r_fit[0] * (nonzero_y ** 2) + r_fit[1] * nonzero_y + r_fit[2] + self._window_width)) &
                      ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high))).nonzero()[0]
            # If we're above the pixel threshold, we consider the pixels in this window to be valid
            if len(lw_ind) > self._pixel_thr:
                left_lane_ind.append(lw_ind)
            if len(rw_ind) > self._pixel_thr:
                right_lane_ind.append(rw_ind)

        # If data is valid, we create lane markers
        if len(left_lane_ind):
            lx = nonzero_x[np.concatenate(left_lane_ind)]
            ly = nonzero_y[np.concatenate(left_lane_ind)]
            left_marker = LaneMarker(img.shape, lx, ly)
            if self._debug:
                dbg_img[ly, lx] = [255, 0, 0]
        else:
            left_marker = None
        if len(right_lane_ind):
            rx = nonzero_x[np.concatenate(right_lane_ind)]
            ry = nonzero_y[np.concatenate(right_lane_ind)]
            right_marker = LaneMarker(img.shape, rx, ry)
            if self._debug:
                dbg_img[ry, rx] = [0, 0, 255]
        else:
            right_marker = None

        return left_marker, right_marker, dbg_img

    def find_lanes_histogram(self, img):
        """
        Finds lane pixels using sliding windows.
        Search starts by the peak of left and right halves
        of the histogram
        """
        # Histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

        # Debug image
        dbg_img = None
        if self._debug:
            dbg_img = np.dstack((img, img, img)) * 255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        lx_base = np.argmax(histogram[:midpoint])
        rx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(img.shape[0] / self._num_windows)

        # Current positions to be updated for each window
        lx_current = lx_base
        rx_current = rx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_ind = []
        right_lane_ind = []

        # Step through the windows one by one
        for window in range(self._num_windows):
            # Defining window boundaries for left and right lanes
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xl_low = lx_current - self._window_width
            win_xl_high = lx_current + self._window_width
            win_xr_low = rx_current - self._window_width
            win_xr_high = rx_current + self._window_width
            # Select active pixels within window boundaries
            active_left_ind = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                               (nonzero_x >= win_xl_low) & (nonzero_x < win_xl_high)).nonzero()[0]
            active_right_ind = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_xr_low) & (nonzero_x < win_xr_high)).nonzero()[0]
            # If we're above the pixel threshold, we consider the pixels in this window
            # valid and center the next window based on the median of the horizontal pixel coordinates
            if len(active_left_ind) > self._pixel_thr:
                left_lane_ind.append(active_left_ind)
                lx_current = np.int(np.median(nonzero_x[active_left_ind]))
            if len(active_right_ind) > self._pixel_thr:
                right_lane_ind.append(active_right_ind)
                rx_current = np.int(np.median(nonzero_x[active_right_ind]))
            if self._debug:
                cv2.rectangle(dbg_img, (win_xl_low, win_y_low), (win_xl_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(dbg_img, (win_xr_low, win_y_low), (win_xr_high, win_y_high), (0, 255, 0), 2)

        # If data is valid, we create lane markers
        if len(left_lane_ind):
            left_lane_ind = np.concatenate(left_lane_ind)
            lx = nonzero_x[left_lane_ind]
            ly = nonzero_y[left_lane_ind]
            left_marker = LaneMarker(img.shape, lx, ly)
            if self._debug:
                dbg_img[ly, lx] = [255, 0, 0]
        else:
            left_marker = None

        if len(right_lane_ind):
            right_lane_ind = np.concatenate(right_lane_ind)
            rx = nonzero_x[right_lane_ind]
            ry = nonzero_y[right_lane_ind]
            right_marker = LaneMarker(img.shape, rx, ry)
            if self._debug:
                dbg_img[ry, rx] = [0, 0, 255]
        else:
            right_marker = None

        return left_marker, right_marker, dbg_img
