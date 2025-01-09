from astropy.table import Table


def solve_astrometry(detection_tbl: Table,
                     ra_initial: float or None,
                     dec_initial: float or None,
                     search_radius: float or None,
                     width: int,
                     height: int,
                     pixel_scale: float,
                     n_source_max: int):
    """
    Find astrometric solution from a table of detected sources.
    :param detection_tbl:  Table of detected sources.
    :param ra_initial: Initial guess for the right ascension of the field center.
    :param dec_initial: Initial guess for the declination of the field center.
    :param search_radius: Search radius for the astrometric solution.
    :param width: Width of the image in pixels.
    :param height: Height of the image in pixels.
    :param pixel_scale: Pixel scale of the image in arcseconds per pixel.
    :param n_source_max: Maximum number of sources to use for the astrometric solution.
    :return:
    """
    pass
