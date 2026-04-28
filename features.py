class RectangleRegion:
    """
    Axis-aligned rectangle in (column, row) = (x, y) coordinates.

      x:      leftmost column (0-indexed)
      y:      topmost row     (0-indexed)
      width:  column-span
      height: row-span

    Looked up against a *padded* summed-area table (see `utils.integral_image`)
    of shape (H+1, W+1) whose first row and column are zero — that lets every
    rectangle sum reduce to four unconditional reads.
    """

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def compute_region(self, ii, scale=1.0, ox=0, oy=0):
        """
        Sum of pixels in this region against padded integral image `ii`.

        Args:
            ii: padded integral image, shape (H+1, W+1), uint32-ish.
            scale: multiplier applied to (x, y, width, height) — used by
                multi-scale sliding-window inference.
            ox, oy: window-origin offset (column, row) into the full-image
                integral image. Zero when `ii` is the per-window II.
        """
        x1 = ox + int(self.x * scale)
        y1 = oy + int(self.y * scale)
        x2 = ox + int((self.x + self.width) * scale)
        y2 = oy + int((self.y + self.height) * scale)
        return (int(ii[y2, x2]) - int(ii[y1, x2])
                - int(ii[y2, x1]) + int(ii[y1, x1]))


class HaarFeature:
    def __init__(self, positive_regions, negative_regions):
        self.positive_regions = positive_regions  # White
        self.negative_regions = negative_regions  # Black

    def compute_value(self, ii, scale=1.0, ox=0, oy=0):
        """Haar feature value: sum(negative) - sum(positive)."""
        sum_pos = sum(r.compute_region(ii, scale, ox, oy) for r in self.positive_regions)
        sum_neg = sum(r.compute_region(ii, scale, ox, oy) for r in self.negative_regions)
        return sum_neg - sum_pos
