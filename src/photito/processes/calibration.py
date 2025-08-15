from astropy.io import fits
import numpy as np
import logging
import ccdproc as ccdp
from astropy.nddata import CCDData
from astropy.stats import mad_std
from ..image_sets import BiasSet
import astropy.units as u

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

def combine_bias_ccdproc(files: list, output: str, mem_limit=32e9,
                         sigma_clip: bool = True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                         combine_method='average', dtype=np.float32):
    """Combine bias frames using ccdproc."""
    # Read bias frames
    bias_frames = [ccdp.CCDData.read(file, unit='adu') for file in files]
    # check all frames have the same gain
    gains = [frame.meta.get('cam-gain', 1.0) for frame in bias_frames]
    if len(set(gains)) > 1:
        logging.error('Bias frames have different gains. Please ensure all bias frames have the same gain.')
        raise ValueError(f'Bias frames have different gains: {set(gains)}.')
    # Combine bias frames
    bias = ccdp.combine(bias_frames, method=combine_method, unit='adu',
                        sigma_clip=sigma_clip, sigma_clip_low_thresh=sigma_clip_low_thresh,
                        sigma_clip_high_thresh=sigma_clip_high_thresh,
                        mem_limit=mem_limit, dtype=dtype)
    # mask values which have saturation or are NaN
    mask = np.isnan(bias.data) | (bias.data <= 0) | (bias.data >= 65535)  # Assuming 16-bit data
    logging.info(f'Masking {np.sum(mask)} pixels in combined bias frame.')
    bias.mask = mask
    # Save combined bias
    bias.meta['cam-gain'] = gains[0]  # Use gain from first frame if available
    bias.meta['combined'] = True
    bias.meta['n_comb'] = len(files)
    bias.meta['comb_m'] = combine_method
    bias.meta['sigclip'] = sigma_clip
    if sigma_clip:
        bias.meta['clip_l'] = sigma_clip_low_thresh
        bias.meta['clip_h'] = sigma_clip_high_thresh
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

    if bias is not None:
        master_bias = ccdp.CCDData.read(bias, unit='adu')
    for file in files:
        dark = ccdp.CCDData.read(file, unit='adu')
        # mask saturated pixels here (as they won't be saturated after bias subtraction)
        mask = np.isnan(dark.data) | (dark.data <= 0) | (dark.data >= 65535)  # Assuming 16-bit data
        logging.info(f'Masking {np.sum(mask)} pixels in dark frame {file} before bias subtraction.'
                     f' {np.sum(mask)/ dark.data.size:.4%} of pixels masked.')
        if bias is not None:
            if dark.meta['cam-gain'] != master_bias.meta['cam-gain']:
                logging.warning(f'Gain mismatch between dark and bias frames: {file} and {bias}.')
            dark = ccdp.subtract_bias(dark, master_bias)
            dark.meta['bias_sub'] = True
            dark.meta['bias_file'] = bias.split('/')[-1]
        dark.meta['calibrated'] = True
        if dark.mask is not None:
            dark.mask = mask | dark.mask  # Combine masks
        else:
            dark.mask = mask

        # Save calibrated dark frame
        dark.write(output_dir + '/' + file.split('/')[-1], overwrite=True)

def combine_darks_ccdproc(files: list, output: str, validate=True, mem_limit=32e9,
                          sigma_clip: bool = True,
                          sigma_clip_low_thresh=5,
                          sigma_clip_high_thresh=5,
                          combine_method='average',
                          dtype=np.float32,
                          max_value=60000):
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
    :param max_value: Mask pixels with values above this threshold
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
    # mask values which have saturation or are NaN
    mask = np.isnan(dark.data)  | (dark.data >= max_value)  # Assuming 16-bit data
    logging.info(f'Masking {np.sum(mask)} pixels in combined dark frame.')
    if dark.mask is not None:
        dark.mask = mask | dark.mask
    else:
        dark.mask = mask
    # Save combined dark
    dark.meta['combined'] = True
    dark.meta['cam-gain'] = ccdp.CCDData.read(files[0], unit='adu').meta.get('cam-gain', 1.0)  # Use gain from first frame if available
    dark.meta['n_comb'] = len(files)
    dark.meta['comb_m'] = combine_method
    dark.meta['sigclip'] = sigma_clip
    if sigma_clip:
        dark.meta['clip_l'] = sigma_clip_low_thresh
        dark.meta['clip_h'] = sigma_clip_high_thresh
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
        logging.info(f'Calibrating {file}')
        flat = ccdp.CCDData.read(file, unit='adu')
        master_dark = ccdp.CCDData.read(dark, unit='adu')
        if bias is not None:
            master_bias = ccdp.CCDData.read(bias, unit='adu')
            flat = ccdp.subtract_bias(flat, master_bias)
            flat.meta['bias_file'] = bias.split('/')[-1]
            if dark is not None:
                flat: CCDData = ccdp.subtract_dark(flat, master_dark, exposure_time='exptime', exposure_unit=u.s, scale=True)
                flat.meta['dark_file'] = dark.split('/')[-1]
        else:
            flat = ccdp.subtract_dark(flat, master_dark, exposure_time='exptime', exposure_unit=u.s, scale=False)
            flat.meta['dark_file'] = dark.split('/')[-1]
        flat.meta['calibrated'] = True
        logging.info(f"Writing calibrated flat frame {file} to {output_dir + '/' + file.split('/')[-1]}.")
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
    filters = [fits.getval(file, 'FILTER', ext=0) for file in files]
    if len(set(filters)) > 1:
        raise ValueError('Flat frames must have the same filter.')
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
                        sigma_clip_high_thresh=sigma_clip_high_thresh, sigclip_func=np.ma.median,
                        sigma_clip_dev_func=mad_std,
                        mem_limit=mem_limit, dtype=dtype, scale=inv_median)
    # mask values which have saturation or are NaN
    mask = np.isnan(flat.data) | (flat.data <= 0) | (flat.data >= 65535)  # Assuming 16-bit data
    logging.info(f'Masking {np.sum(mask)} pixels in combined flat frame.')
    flat.mask = mask
    # Save combined flat
    flat.meta['combined'] = True
    flat.meta['n_comb'] = len(files)
    flat.meta['comb_m'] = combine_method
    flat.meta['sigclip'] = sigma_clip
    if sigma_clip:
        flat.meta['clip_l'] = sigma_clip_low_thresh
        flat.meta['clip_h'] = sigma_clip_high_thresh
    flat.write(output, overwrite=True)


def calibrate_lights_ccdproc(files:list, output_dir:str,
                             master_bias:str=None, master_dark:str=None, master_flat:str=None):
    """Calibrate light frames using ccdproc.
    :param files: List of light frames.
    :param output_dir: Output folder.
    :param master_bias: Master bias frame location.
    :param master_dark: Master dark frame location.
    :param master_flat: Master flat frame location.
    """
    scale = False
    mask = None
    if master_bias is not None:
        master_bias_data = ccdp.CCDData.read(master_bias, unit='adu')
        scale = True
    if master_dark is not None:
        master_dark_data = ccdp.CCDData.read(master_dark, unit='adu')
    if master_flat is not None:
        master_flat_data = ccdp.CCDData.read(master_flat, unit='adu')
    for file in files:
        image = ccdp.CCDData.read(file, unit='adu')
        mask = np.isnan(image.data) | (image.data <= 0) | (image.data >= 65535)  # Assuming 16-bit data
        logging.info(f'Masking {np.sum(mask)} pixels in light frame {file} before calibration.'
                     f' {np.sum(mask)/ image.data.size:.4%} of pixels masked.')
        if master_bias is not None:
            image: CCDData = ccdp.subtract_bias(image, master_bias_data)
            image.meta['bias_f'] = master_bias.split("/")[-1]
        if master_dark is not None:
            image = ccdp.subtract_dark(image, master_dark_data, exposure_time='exptime', exposure_unit=u.s, scale=scale)
            image.meta['dark_f'] = master_dark.split("/")[-1]
        if master_flat is not None:
            image = ccdp.flat_correct(image, master_flat_data, min_value=0.1, norm_value=1)
            image.meta['flat_f'] = master_flat.split("/")[-1]
        # mask values which were masked in the calibration steps or are NaN
        if image.mask is not None:
            image.mask = mask | image.mask
        else:
            image.mask = mask
        logging.info(f'Masking {np.sum(mask)} pixels in calibrated light frame {file}.')
        image.mask = mask
        # Save calibrated light frame
        image.meta['calib'] = True
        image.write(output_dir + '/' + file.split('/')[-1], overwrite=True)

