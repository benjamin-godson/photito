from ccdproc import ImageFileCollection


class ImageSet(ImageFileCollection):
    """Base class for a set of images. Based on ccdproc.ImageFileCollection."""

    def __init__(self, files: list):
        """Initialize the image set.
        :param files: List of image files.
        """
        super().__init__(filenames=files)


class CalibrationSet(ImageSet):
    """A set of calibration images."""


class BiasSet(CalibrationSet):
    """A set of bias images."""

    def __init__(self, files: list):
        """Initialize the bias set.
        :param files: List of bias files.
        """
        super().__init__(files)
        if not self._verify_imagetyp():
            raise ValueError('All images must be bias frames.')

    def _verify_imagetype(self):
        """Check if all images are bias frames"""
        return all(imagetyp.lower() == 'bias' for imagetyp in self.summary['imagetyp'])
