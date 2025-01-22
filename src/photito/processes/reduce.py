import numpy as np
from concurrent.futures import ThreadPoolExecutor
from photito import config

from astropy.stats import mad_std, sigma_clip
from functools import partial

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

        sections = np.array_split(self.data, n_sections, axis=1)
        if self.mask is not None:
            mask_sections = np.array_split(self.mask, n_sections, axis=1)
        else:
            mask_sections = [None] * n_sections
        if self.uncertainty is not None:
            uncertainty_sections = np.array_split(self.uncertainty, n_sections, axis=1)
        else:
            uncertainty_sections = [None] * n_sections
        if self.scaling is not None:
            scaling_sections = np.array_split(self.scaling, n_sections, axis=1)
        else:
            scaling_sections = [None] * n_sections
        if self.weights is not None:
            weights_sections = np.array_split(self.weights, n_sections, axis=1)
        else:
            weights_sections = [None] * n_sections

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
                                        cenfunc=np.ma.median,)

        stacked_data = np.ma.average(sigma_clipped_data, axis=0, weights=weights)
        return stacked_data.filled(fill_value=np.nan)





