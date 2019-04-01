class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # self.x1 = int(x - 1)
        # self.y1 = int(y - 1)
        # self.x2 = int(x + width - 1)
        # self.y2 = int(y + height - 1)

    def compute_region(self, ii, scale=1.0):
        # D(all) - C(left) - B(top) + A(corner)

        x1 = self.x
        y1 = self.y
        x2 = x1 + self.width - 1
        y2 = y1 + self.height - 1

        S = int(ii[x2, y2])
        if x1 > 0: S -= int(ii[x1-1, y2])
        if y1 > 0: S -= int(ii[x2, y1-1])
        if x1 > 0 and y1 > 0: S += int(ii[x1 - 1, y1 - 1])
        return S  # Due to the use of substraction with unsigned values


class HaarFeature:
    def __init__(self, positive_regions, negative_regions):
        self.positive_regions = positive_regions  # White
        self.negative_regions = negative_regions  # Black

    def compute_value(self, ii, scale=1.0):
        """
        Compute the value of a feature(x,y,w,h) at the integral image
        """

        sum_pos = sum([rect.compute_region(ii, scale) for rect in self.positive_regions])
        sum_neg = sum([rect.compute_region(ii, scale) for rect in self.negative_regions])
        return sum_pos - sum_neg
