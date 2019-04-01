class RectangleRegion:
    def __init__(self, x, y, width, height):
        # self.x = x
        # self.y = y
        # self.width = width
        # self.height = height

        self.x1 = int(x - 1)
        self.y1 = int(y - 1)
        self.x2 = int(x + width - 1)
        self.y2 = int(y + height - 1)

    def compute_region(self, ii, scale=1.0):
        # D(all) - C(left) - B(top) + A(corner)

        w= self.x2-self.x1+1
        h= self.y2-self.y1+1

        # # TODO: Review
        x1 = int(self.x1 * scale)
        y1 = int(self.y1 * scale)
        x2 = x1+int(w * scale - 1)
        y2 = int(h * scale - 1)

        d = ii[x2, y2]
        b = ii[x2, y1] if y1 >= 0 else 0
        c = ii[x1, y2] if x1 >= 0 else 0
        a = ii[x1, y1] if x1 >= 0 and y1 >= 0 else 0

        # d = ii[self.x2, self.y2]
        # b = ii[self.x2, self.y1] if self.y1 >= 0 else 0
        # c = ii[self.x1, self.y2] if self.x1 >= 0 else 0
        # a = ii[self.x1, self.y1] if self.x1 >= 0 and self.y1 >= 0 else 0

        return int(d + a) - int(c + b)  # Due to the use of substraction with unsigned values


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
