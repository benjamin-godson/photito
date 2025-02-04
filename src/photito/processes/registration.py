from spalipy import Spalipy
from astropy.io import fits
import numpy as np
import logging


def align_lights(files: list, output_dir: str, reference: str = None):
    """Align light frames to a reference frame.
    :param files: List of light frames.
    :param output_dir: Output folder.
    :param reference: Reference frame.
    """
    logging.info(f'Aligning {len(files)} light frames.')
    if reference is None:
        reference = files[len(files) // 2]
    logging.info(f'Using {reference} as reference frame.')
    reference_data = fits.getdata(reference)
    reference_data = np.nan_to_num(reference_data)
    for file in files:
        logging.info(f'Aligning {file}.')
        image_data = fits.getdata(file)
        image_data = np.nan_to_num(image_data)
        image_header = fits.getheader(file)
        spalipy = Spalipy(image_data, template_data=reference_data)
        spalipy.align()
        fits.writeto(f'{output_dir}/aligned_{file.split("/")[-1]}', spalipy.aligned_data,
                     header=image_header, overwrite=True)



