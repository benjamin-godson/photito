import os
import numpy as np
import pytest
from photito.processes.calibration import combine_bias, combine_bias_ccdproc
from photito.image_sets import BiasSet
from astropy.io import fits
def test_files_exist(path='tests/data/bias'):
    """Check if test files are present."""
    files = os.listdir(path)
    assert 'bias_gain0_bin1-i-20240826040143760.fits' in files


def test_create_bias_set(path='tests/data/bias'):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    bias_set = BiasSet(files)
    assert (bias_set.summary['file'] == files).all()
    assert (imagetyp.lower() == 'bias' for imagetyp in bias_set.summary['imagetyp'])

def test_bias_combine_method(path='tests/data/bias', output='tests/data/bias_combined.fits'):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    bias_set = BiasSet(files)
    bias_set.combine(output)
    assert os.path.exists(output)
    with fits.open(output) as hdul:
        assert hdul[0].header['BITPIX'] == -32
        assert hdul[0].header['NAXIS'] == 2
        assert hdul[0].data.shape == (10656, 14208)
    os.remove(output)

@pytest.mark.skip(reason="Slow test")
def test_combine_bias(path='tests/data/bias', output='tests/data/bias_combined.fits'):
    """Check if combine_bias works."""
    files = [os.path.join(path, f) for f in os.listdir(path)]
    combine_bias(files, output)
    assert os.path.exists(output)
    with fits.open(output) as hdul:
        assert hdul[0].header['BITPIX'] == -32
        assert hdul[0].header['NAXIS'] == 2
        assert hdul[0].data.shape == (10656, 14208)
    os.remove(output)


@pytest.mark.skip(reason="Slow test")
def test_combine_bias_ccdproc(path='tests/data/bias', output='tests/data/bias_combined.fits'):
    """Check if combine_bias_ccdproc works."""
    files = [os.path.join(path, f) for f in os.listdir(path)]
    combine_bias_ccdproc(files, output)
    assert os.path.exists(output)
    os.remove(output)
