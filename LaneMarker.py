import numpy as np

class LaneMarker:
    """
    Stores all info that compose a lane marker
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
        self._poly_fit_px = None
        self._poly_fit_m = None

    def poly_fit_px(self):
        if self._poly_fit_px is None:
            self._poly_fit_px = np.polyfit(self._y_values, self._x_values, 2)
        return self._poly_fit_px

    def poly_fit_m(self):
        if self._poly_fit_m is None:
            self._poly_fit_m = np.polyfit(self._y_values * LaneMarker.ym_per_px,
                                          self._x_values * LaneMarker.xm_per_px, 2)
        return self._poly_fit_m

    def curvature_radius(self):
        fit = self.poly_fit_m()
        radius = ((1 + ((2 * fit[0] * self._img_shape[0] * LaneMarker.ym_per_px) + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        return radius
