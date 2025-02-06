from spalipy import Spalipy
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import CCDData
import sep
import numpy as np
import logging
from multiprocessing import Pool

logger = logging.getLogger(__name__)

def align_lights(files: list, output_dir: str, reference: str = None, n_processes: int = 1):
    """Align light frames to a reference frame.
    :param files: List of light frames.
    :param output_dir: Output folder.
    :param reference: Reference frame.
    :param n_processes: Number of processes to use.
    """
    logger.info(f'Aligning {len(files)} light frames.')
    if reference is None:
        reference = files[len(files) // 2]
    logger.info(f'Using {reference} as reference frame.')
    reference_data = fits.getdata(reference)
    reference_data = np.nan_to_num(reference_data)
    reference_detections = reference_extraction(reference_data)
    if n_processes > 1:
        with Pool(n_processes) as pool:
            pool.starmap(align_light_thread, [(file, reference_data, output_dir) for file in files])
    else:
        for file in files:
            align_light_thread(file, reference_data, output_dir)


def align_light_thread(file: str, reference_detections: Table, output_dir: str):
    """Align a light frame to a reference frame."""
    logger.info(f'Aligning {file}.')
    image_data = fits.getdata(file)
    mask = np.isnan(image_data)
    mask += image_data <= 0
    image_data = np.nan_to_num(image_data)
    image_data = np.clip(image_data, 0, None)
    image_header = fits.getheader(file)
    spalipy = Spalipy(image_data, template_det=reference_detections, source_mask=mask)
    spalipy.align()
    combined_data = CCDData(data=spalipy.aligned_data, mask=spalipy.aligned_mask, header=image_header, unit='adu')
    combined_data.write(f'{output_dir}/aligned_{file.split("/")[-1]}', overwrite=True)

def reference_extraction(data: np.ndarray, threshold: float = 1.5):
    """Extract sources from a reference frame."""
    if data.dtype.byteorder == '>':
        data = data.byteswap().newbyteorder()
    background = sep.Background(data)
    data_sub = data - background

    detections = sep.extract(data_sub, threshold, err=background.rms())
    detections = Table(detections)
    detections['fwhm'] = 2.0 * (np.log(2) * (detections["a"] ** 2.0 + detections["b"] ** 2.0)) ** 0.5
    return detections

