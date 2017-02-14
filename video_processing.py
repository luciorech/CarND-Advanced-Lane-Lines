import argparse
import pickle
import numpy as np
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from LaneFinder import LaneFinder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Driving Lane Detection')
    parser.add_argument('--input', default='./project_video.mp4', help='Path to input video file.')
    parser.add_argument('--output', default='./output_videos/project_video.mp4', help='Path to output video file.')
    parser.add_argument('--calibration_file', default='./camera_cal/calibration.p', help='Camera calibration params')
    parser.add_argument('--start', default=0, type=float)
    parser.add_argument('--end', default=None, type=float)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.set_defaults(debug=False)

    args = parser.parse_args()

    with open(args.calibration_file, 'rb') as pfile:
        cal_dict = pickle.load(pfile)
        mtx = cal_dict["mtx"]
        dist = cal_dict["dist"]

    finder = LaneFinder(kernel_size=3,
                        cam_mtx=mtx,
                        cam_dist=dist,
                        s_thr=(100, 255),
                        l_thr=50,
                        sobel_thr=(20, 180),
                        window_width=75,
                        num_windows=10,
                        pixel_thr=50,
                        debug=args.debug)

    start = float(args.start)
    end = float(args.end) if args.end is not None else None
    in_video = VideoFileClip(args.input).subclip(t_start=start, t_end=end)
    out_video = in_video.fl_image(finder.process_frame)
    out_video.write_videofile(args.output, audio=False)

    if args.debug:
        fps = in_video.fps
        print("fps = %s" % fps)
        frame_cnt = 0
        frame_info = "L radius = {0:4.1f} (pts = {1:d}), R radius = {2:4.1f} (pts = {3:d}) - Offset = {4:1.2f}, Failed : {5}"
        fit_info = "L fit = {0}, R fit = {1}"
        for frame in finder.debug_log():
            print(frame_info.format(frame['left'].curvature_radius(),
                                    len(frame['left'].x_values()),
                                    frame['right'].curvature_radius(),
                                    len(frame['right'].x_values()),
                                    frame['offset'],
                                    frame['failed']))
            print(fit_info.format(frame['left'].poly_fit_px(),
                                  frame['right'].poly_fit_px()))

            frame_img = in_video.get_frame(frame_cnt / fps)
            f, ax = plt.subplots(4, 3, figsize=(15, 10))
            f.subplots_adjust(hspace=0.3)
            ax[0, 0].imshow(frame_img)
            ax[0, 0].set_title('Video', fontsize=10)
            ax[0, 1].imshow(frame['undistorted'], cmap='gray')
            ax[0, 1].set_title('Undistorted', fontsize=10)
            ax[0, 2].imshow(frame['result'])
            ax[0, 2].set_title('Pipeline', fontsize=10)

            ax[1, 0].imshow(frame['s'], cmap='gray')
            ax[1, 0].set_title('S (HLS)', fontsize=10)
            ax[1, 1].imshow(frame['l'], cmap='gray')
            ax[1, 1].set_title('L (HLS)', fontsize=10)
            ax[1, 2].imshow(frame['red'], cmap='gray')
            ax[1, 2].set_title('R (RGB)', fontsize=10)

            ax[2, 0].imshow(frame['x_gradient'], cmap='gray')
            ax[2, 0].set_title('Grad X', fontsize=10)
            ax[2, 1].imshow(frame['y_gradient'], cmap='gray')
            ax[2, 1].set_title('Grad Y', fontsize=10)
            ax[2, 2].imshow(frame['s_binary'], cmap='gray')
            ax[2, 2].set_title('S Binary', fontsize=10)

            ax[3, 1].imshow(frame['combined'], cmap='gray')
            ax[3, 1].set_title('Combined', fontsize=10)
            pipeline_img = frame['warped']
            ax[3, 2].imshow(pipeline_img, cmap='gray')
            left_fit = frame['left'].poly_fit_px()
            right_fit = frame['right'].poly_fit_px()
            plot = np.linspace(0, pipeline_img.shape[0]-1, pipeline_img.shape[0])
            lx = left_fit[0] * plot ** 2 + left_fit[1] * plot + left_fit[2]
            lx = np.clip(lx, 0, pipeline_img.shape[1] - 1)
            rx = right_fit[0] * plot ** 2 + right_fit[1] * plot + right_fit[2]
            rx = np.clip(rx, 0, pipeline_img.shape[1] - 1)
            ax[3, 2].plot(lx, plot, color='yellow')
            ax[3, 2].plot(rx, plot, color='yellow')
            ax[3, 2].set_title('Warped', fontsize=10)

            plt.show(block=True)
            frame_cnt += 1

