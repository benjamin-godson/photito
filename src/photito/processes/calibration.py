from astropy.io import fits
import numpy as np
import logging
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

def combine_bias_ccdproc(files: list, output: str):
    """Combine bias frames using ccdproc."""
    # Read bias frames
    bias_frames = [ccdp.CCDData.read(file, unit='adu') for file in files]
    # Combine bias frames
    bias = ccdp.combine(bias_frames, method='median')
    # Save combined bias
    bias.write(output, overwrite=True)

def combine_darks_ccdproc(files: list, output: str, validate=True, mem_limit=32e9):
    """Combine dark frames using ccdproc."""
    for file in files:
        image_type = fits.getval(file, 'IMAGETYP', ext=0).lower()
        if image_type != 'dark':
            if validate:
                raise ValueError(f'Image {file} is not a dark frame.')
            else:
                logging.warning(f'Image {file} is not a dark frame.')
    # Combine dark frames
    dark = ccdp.combine(files, method='average', unit='adu',
                        sigma_clip=True, mem_limit=mem_limit, dtype=np.float32)
    # Save combined dark
    dark.meta['combined'] = True
    dark.write(output, overwrite=True)
