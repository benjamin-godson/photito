from ccdproc import ImageFileCollection
from astropy.io import fits
import numpy as np
from photito.processes.reduce import Stacker

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
        if not self._verify():
            raise ValueError('All images must be bias frames.')

    def _verify(self):
        """Check if all images are bias frames at the same gain and with matching dimensions."""
        correct_type = all(imagetyp.lower() == 'bias' for imagetyp in self.summary['imagetyp'])
        same_gain = all(gain == self.summary['cam-gain'][0] for gain in self.summary['cam-gain'])
        same_dimensions = all((width, height) == (self.summary['naxis1'][0], self.summary['naxis2'][0])
                              for width, height in zip(self.summary['naxis1'], self.summary['naxis2']))
        return correct_type and same_gain and same_dimensions

    def combine(self, output: str, n_sections=100):
        """Combine bias frames."""
        # Read bias frames
        bias_frames = np.stack([fits.open(file)[0].data for file in self.files], axis=0)
        combined_bias = Stacker(data=bias_frames, n_sections=n_sections).threaded_sigma_clip()
        # Save combined bias
        hdu = fits.PrimaryHDU(combined_bias)
        hdu.writeto(output, overwrite=True)


