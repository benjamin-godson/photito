from astropy.io import fits
import numpy as np
import logging
import ccdproc as ccdp
from ..image_sets import BiasSet

def inv_median(a):
    return 1 / np.median(a)

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

def calibrate_darks_ccdproc(files: list, output_dir: str, bias: str = None, mem_limit=32e9):
    """Calibrate dark frames using ccdproc.
    :param files: List of dark frames.
    :param output_dir: Output folder.
    :param bias: Master bias frame location.
    :param mem_limit: Memory limit for the operation.
    """
    for file in files:
        image_type = fits.getval(file, 'IMAGETYP', ext=0).lower()
        if image_type != 'dark':
            raise ValueError(f'Image {file} is not a dark frame.')
    # Check if all dark frames have the same exposure time
    # Calibrate dark frames
    for file in files:
        dark = ccdp.CCDData.read(file, unit='adu')
        if bias is not None:
            master_bias = ccdp.CCDData.read(bias, unit='adu')
            if dark.meta['cam-gain'] != master_bias.meta['cam-gain']:
                logging.warning(f'Gain mismatch between dark and bias frames: {file} and {bias}.')
            dark = ccdp.subtract_bias(dark, master_bias)
            dark.meta['bias_sub'] = True
            dark.meta['bias_file'] = bias.split('/')[-1]
        dark.meta['calibrated'] = True
        dark.write(output_dir + '/' + file.split('/')[-1], overwrite=True)

def combine_darks_ccdproc(files: list, output: str, validate=True, mem_limit=32e9,
                          sigma_clip: bool = True,
                          sigma_clip_low_thresh=5,
                          sigma_clip_high_thresh=5,
                          combine_method='average',
                          dtype=np.float32):
    """Combine dark frames using ccdproc.
    :param files: List of dark frames.
    :param output: Output file name.
    :param validate: Error if the images are not all dark frames.
    :param mem_limit: Memory limit for the operation.
    :param sigma_clip: Use sigma clipping.
    :param sigma_clip_low_thresh: Low threshold for sigma clipping.
    :param sigma_clip_high_thresh: High threshold for sigma clipping.
    :param combine_method: Method for combining the frames.
    :param dtype: Data type for the output.
    """
    for file in files:
        image_type = fits.getval(file, 'IMAGETYP', ext=0).lower()
        if image_type != 'dark':
            if validate:
                raise ValueError(f'Image {file} is not a dark frame.')
            else:
                logging.warning(f'Image {file} is not a dark frame.')
    # Combine dark frames
    dark = ccdp.combine(files, method=combine_method, unit='adu',
                        sigma_clip=sigma_clip, sigma_clip_low_thresh=sigma_clip_low_thresh,
                        sigma_clip_high_thresh=sigma_clip_high_thresh,
                        mem_limit=mem_limit, dtype=dtype)
    # Save combined dark
    dark.meta['combined'] = True
    dark.meta['combine_method'] = combine_method
    dark.meta['sigma_clip'] = sigma_clip
    if sigma_clip:
        dark.meta['sigma_clip_low_thresh'] = sigma_clip_low_thresh
        dark.meta['sigma_clip_high_thresh'] = sigma_clip_high_thresh
    dark.write(output, overwrite=True)

def calibrate_flats_ccdproc(files: list, output_dir: str, dark: str = None, bias: str = None, mem_limit=32e9):
    """Calibrate flat frames using ccdproc. Only supply a bias frame if using dark scaling.
    :param files: List of flat frames.
    :param output_dir: Output folder.
    :param bias: Master bias frame location.
    :param dark: Master dark frame location.
    :param mem_limit: Memory limit for the operation.
    """
    for file in files:
        image_type = fits.getval(file, 'IMAGETYP', ext=0).lower()
        if image_type != 'flat':
            raise ValueError(f'Image {file} is not a flat frame.')
    # Calibrate flat frames
    for file in files:
        flat = ccdp.CCDData.read(file, unit='adu')
        master_dark = ccdp.CCDData.read(dark, unit='adu')
        if bias is not None:
            master_bias = ccdp.CCDData.read(bias, unit='adu')
            flat = ccdp.subtract_bias(flat, master_bias)
            flat.meta['bias_file'] = bias.split('/')[-1]
            if dark is not None:
                flat = ccdp.subtract_dark(flat, master_dark, exposure_time='exptime', exposure_unit='s', scale=True)
                flat.meta['dark_file'] = dark.split('/')[-1]
        else:
            flat = ccdp.subtract_dark(flat, master_dark, exposure_time='exptime', exposure_unit='s', scale=False)
            flat.meta['dark_file'] = dark.split('/')[-1]
        flat.meta['calibrated'] = True
        flat.write(output_dir + '/' + file.split('/')[-1], overwrite=True)

def combine_flats_ccdproc(files: list, output: str, validate=True, mem_limit=32e9,
                          sigma_clip: bool = True,
                          sigma_clip_low_thresh=5,
                          sigma_clip_high_thresh=5,
                          combine_method='average',
                          dtype=np.float32):
    """Combine flat frames using ccdproc.
    :param files: List of flat frames.
    :param output: Output file name.
    :param validate: Error if the images are not all flat frames.
    :param mem_limit: Memory limit for the operation.
    :param sigma_clip: Use sigma clipping.
    :param sigma_clip_low_thresh: Low threshold for sigma clipping.
    :param sigma_clip_high_thresh: High threshold for sigma clipping.
    :param combine_method: Method for combining the frames.
    :param dtype: Data type for the output.
    """
    for file in files:
        image_type = fits.getval(file, 'IMAGETYP', ext=0).lower()
        if image_type != 'flat':
            if validate:
                raise ValueError(f'Image {file} is not a flat frame.')
            else:
                logging.warning(f'Image {file} is not a flat frame.')
    # Combine flat frames
    flats = [ccdp.CCDData.read(file, unit='adu') for file in files]
    flat = ccdp.combine(flats, method=combine_method, unit='adu',
                        sigma_clip=sigma_clip, sigma_clip_low_thresh=sigma_clip_low_thresh,
                        sigma_clip_high_thresh=sigma_clip_high_thresh,
                        mem_limit=mem_limit, dtype=dtype, scale=inv_median)
    # Save combined flat
    flat.meta['combined'] = True
    flat.meta['combine_method'] = combine_method
    flat.meta['sigma_clip'] = sigma_clip
    if sigma_clip:
        flat.meta['sigma_clip_low_thresh'] = sigma_clip_low_thresh
        flat.meta['sigma_clip_high_thresh'] = sigma_clip_high_thresh
    flat.write(output, overwrite=True)
