from astropy.io import fits
import numpy as np

import ccdproc as ccdp
from ..image_sets import BiasSet

def combine_bias(files: list, output: str):
    """Combine bias frames."""
    # Read bias frames
    bias_frames = [fits.open(file)[0].data for file in files]
    # Combine bias frames
    bias = np.median(bias_frames, axis=0, out=np.empty_like(bias_frames[0], dtype=np.float32))
    # Save combined bias
    hdu = fits.PrimaryHDU(bias)
    hdu.writeto(output, overwrite=True)

def combine_bias_ccdproc(files:list, output:str):
    """Combine bias frames using ccdproc."""
    # Read bias frames
    bias_frames = [ccdp.CCDData.read(file, unit='adu') for file in files]
    # Combine bias frames
    bias = ccdp.combine(bias_frames, method='median')
    # Save combined bias
    bias.write(output, overwrite=True)