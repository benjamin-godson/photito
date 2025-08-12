import numpy as np
from concurrent.futures import ThreadPoolExecutor
from photito import config
import ccdproc as ccdp
from astropy.stats import mad_std, sigma_clip
from astropy.io import fits
from functools import partial
from itertools import repeat
import logging


class Stacker:
    def __init__(self,
                 data: np.ndarray,
                 mask: np.ndarray = None,
                 uncertainty: np.ndarray = None,
                 n_sections: int = 1,
                 scaling: np.ndarray = None,
                 weights: np.ndarray = None,
                 ):
        if n_sections < 1:
            raise ValueError('n_sections must be greater than 0.')
        self.data = data
        if mask is not None:
            self.mask = mask
        else:
            self.mask = np.zeros_like(self.data)
        if uncertainty is not None:
            self.uncertainty = uncertainty
        else:
            self.uncertainty = np.ones_like(self.data)
        if scaling is not None:
            self.scaling = scaling
        else:
            self.scaling = np.ones_like(self.data)

        self.uncertainty = uncertainty
        self.scaling = scaling
        self.weights = weights
        self.n_sections = n_sections

    def threaded_sigma_clip(self, sigma: float = 5.0, maxiters: int = 5):
        """Average the data with sigma clipping."""
        if self.n_sections == 1:
            return self.data
        if self.uncertainty is None:
            self.uncertainty = np.ones_like(self.data, dtype=float)
        if self.scaling is None:
            self.scaling = np.ones_like(self.data, dtype=float)
        if self.weights is None:
            self.weights = np.ones_like(self.data, dtype=float)

        n_sections = min(self.n_sections, self.data.shape[1])
        logging.info(f'Stacking data in {n_sections} sections.')

        sections = np.array_split(self.data, n_sections, axis=1)
        if self.mask is not None:
            mask_sections = np.array_split(self.mask, n_sections, axis=1)
        else:
            mask_sections = repeat(None)
        if self.uncertainty is not None:
            uncertainty_sections = np.array_split(self.uncertainty, n_sections, axis=1)
        else:
            uncertainty_sections = repeat(None)
        if self.scaling is not None:
            scaling_sections = np.array_split(self.scaling, n_sections, axis=1)
        else:
            scaling_sections = repeat(None)
        if self.weights is not None:
            weights_sections = np.array_split(self.weights, n_sections, axis=1)
        else:
            weights_sections = repeat(None)

        with ThreadPoolExecutor(max_workers=config.n_processes) as executor:
            results = list(executor.map(self._average_sigma_clip,
                                        sections,
                                        mask_sections,
                                        uncertainty_sections,
                                        scaling_sections,
                                        weights_sections,
                                        [sigma] * n_sections,
                                        [maxiters] * n_sections))
            return np.concatenate(results, axis=1).astype(np.float32)

    def _average_sigma_clip(self,
                            data: np.ndarray = None,
                            mask: np.ndarray = None,
                            uncertainty: np.ndarray = None,
                            scaling: np.ndarray = None,
                            weights: np.ndarray = None,
                            sigma: float = 5.0,
                            maxiters: int = 5):
        """Average the data in a section with sigma clipping."""
        if mask is None:
            mask = np.ma.nomask
        data = np.ma.array(data, mask=mask)

        sigma_clipped_data = sigma_clip(data=data,
                                        sigma=sigma,
                                        maxiters=maxiters,
                                        stdfunc=mad_std,
                                        cenfunc=np.ma.median)

        stacked_data = np.ma.average(sigma_clipped_data, axis=0, weights=weights)
        return stacked_data.filled(fill_value=np.nan)


def combine_lights_ccdproc(files: list, output: str, mem_limit=32e9,
                           sigma_clip: bool = True,
                           sigma_clip_low_thresh=3,
                           sigma_clip_high_thresh=3,
                           combine_method='average',
                           dtype=np.float32):
    """Combine light frames.
    :param files: List of light frames.
    :param output: Output file name.
    :param mem_limit: Memory limit for the operation.
    :param sigma_clip: Use sigma clipping.
    :param sigma_clip_low_thresh: Low threshold for sigma clipping.
    :param sigma_clip_high_thresh: High threshold for sigma clipping.
    :param combine_method: Method for combining the frames.
    :param dtype: Data type for the output.
    """
    # Combine light frames
    master = ccdp.combine(files, method=combine_method, unit='adu',
                          sigma_clip=sigma_clip, sigma_clip_low_thresh=sigma_clip_low_thresh,
                          sigma_clip_high_thresh=sigma_clip_high_thresh,
                          mem_limit=mem_limit, dtype=dtype)
    # Save combined light
    master.meta['combined'] = True
    master.meta['n_combine'] = len(files)
    master.meta['combine_method'] = combine_method
    master.meta['sigma_clip'] = sigma_clip
    if sigma_clip:
        master.meta['sigma_clip_low_thresh'] = sigma_clip_low_thresh
        master.meta['sigma_clip_high_thresh'] = sigma_clip_high_thresh
    master.write(output, overwrite=True)
