import numpy as np


class LaneMarker:
    """
    Stores info related to a lane marker
    """
    ym_per_px = 30 / 720  # meters per pixel in y dimension
    xm_per_px = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self,
                 img_shape,
                 x_values,
                 y_values):
        self._img_shape = img_shape
        self._x_values = x_values
        self._y_values = y_values
        self._fit_px = None
        self._fit_m = None
        self._roc = None

    def x_values(self):
        return self._x_values

    def y_values(self):
        return self._y_values

    def img_shape(self):
        return self._img_shape

    def poly_fit_px(self):
        if self._fit_px is None:
            self._fit_px = np.polyfit(self._y_values, self._x_values, 2)
        return self._fit_px

    def poly_fit_m(self):
        if self._fit_m is None:
            self._fit_m = np.polyfit(self._y_values * LaneMarker.ym_per_px, self._x_values * LaneMarker.xm_per_px, 2)
        return self._fit_m

    def curvature_radius(self):
        if self._roc is None:
            fit = self.poly_fit_m()
            self._roc = ((1 + ((2 * fit[0] * self._img_shape[0] * LaneMarker.ym_per_px) + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
            if fit[0] < 0:
                self._roc = -self._roc
        return self._roc

    def x_px_pos(self, y):
        fit = self.poly_fit_px()
        x = fit[0] * y ** 2 + fit[1] * y + fit[2]
        return x
