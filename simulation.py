from math import cos, sin, pi, exp, atan
import tqdm
import numpy as np
import scipy.integrate as integ

interval_Y = [0, 7]
interval_X = [-3, 3]
résolution_Y = 1200
résolution_X = int(résolution_Y*(interval_X[1]-interval_X[0])/(interval_Y[1]-interval_Y[0]))

Froude_a_calculer = [2]


def pressure_field(teta, FR):
    K0_inv = (FR * cos(teta))**2
    return exp(- 1 / (2 * pi * K0_inv)**2)


def fonction_a_integrer(teta: float, x_tilde: float, y_tilde: float, froude_nbr: float) -> float:
    sin_numerator = 2 * pi * (cos(teta) * x_tilde - sin(teta) * y_tilde)
    sin_denominator = cos(teta)**2

    numerator = sin(sin_numerator / sin_denominator)
    denominator = cos(teta)**4

    return pressure_field(teta, froude_nbr) * numerator / denominator


def surface_displacement(x_tilde: float, y_tilde: float, froude_nbr: float) -> float:
    value, _ = integ.quad(fonction_a_integrer,
                          a=-pi / 2,
                          b=pi / 2,
                          args=(x_tilde, y_tilde, froude_nbr),
                          limit=250,
                          epsabs=1e-4,
                          epsrel=1e-4,)
    return -value


def wake_simulation(
        froude_nbr: float,
        *,
        image_height: int = 600,
        image_width: int = None,
        max_x_tilde: float = 3,
        max_y_tilde: float = 10,
        use_mask: bool = True,
        mask: np.ndarray = None,
        progress_bar: tqdm.tqdm = tqdm.tqdm(),
        vertical_offset: float = 0.05,
        exclude_top_margin: bool = True
) -> np.ndarray:
    """
    Compute the z elevation and output an image representing it.

    Parameters
    ----------
    froude_nbr : float
        froude number for the simulation
    image_height : int, optional
        the height of the output image. If set to None, will be computed
        using `image_width`, `max_x_tilde`, `max_y_tilde`, by default 600
    image_width : int, optional
        the width of the output image. If set to None, will be computed
        using `image_height`, `max_x_tilde`, `max_y_tilde`, by default None
    max_x_tilde : float, optional
        the max X coordinate for the simulation. If set to None, will be
        computed using `image_width`, `image_height`, `max_y_tilde`, by default 3
    max_y_tilde : float, optional
        the max X coordinate for the simulation. If set to None, will be
        computed using `image_width`, `image_height`, `max_x_tilde`, by default 10
    use_mask : bool, optional
        whether to use a mask to avoid computing null area. Using a mask
        for low froude number can truncate non-null areas, by default True
    mask : np.ndarray, optional
        a custom mask as a bool nparray can be provided to compute only
        the interesting areas, by default None.
    progress_bar : tqdm.tqdm, optional
        display a progress bar for the progress of the simulation.It is useful
        as the simulation can take a significant time, by default tqdm.tqdm()
    vertical_offset : float, optional
        relative space between the top and the singularity, by default 0.05
    exclude_top_margin : bool, optional
        whether or not to compute what is in the top margin defined by
        vertical_offset, by default True

    Returns
    -------
    np.ndarray
        2d-matrix representing the z-elevation image
    """

    parametres_combination = (image_height, image_width, max_x_tilde, max_y_tilde)
    if sum([x is not None for x in parametres_combination]) != 1:
        raise ValueError("exactly one of the parametres `image_height`, `image_width`,"
                         "`x_tilde_range`, `y_tilde_range` should be None. The others"
                         "must have a value.")

    if image_height is None:
        image_height = image_width / max_x_tilde * max_y_tilde

    if image_width is None:
        image_width = image_height / max_y_tilde * max_x_tilde

    if max_x_tilde is None:
        max_x_tilde = image_width / image_height * max_y_tilde

    if max_y_tilde is None:
        max_y_tilde = image_height / image_width * max_x_tilde

    # to convert image coordinates to simulation coordinates.
    # the image will typically be 400*600 whereas the simulation
    # will be 6 * 10
    subpixel_nbr = image_height / max_y_tilde

    # to move the singularity further down
    y_offset = int(image_height * vertical_offset)

    if mask is None:
        mask = compute_mask(
            image_height,
            image_width,
            y_offset=y_offset,
            blank_mask=(not use_mask),
            exclude_top_margin=exclude_top_margin
        )

    if progress_bar is not None:
        pixel_count = mask.sum()
        progress_bar.total = pixel_count

    half_width = int(image_width / 2)
    image = np.zeros((image_height, half_width), dtype=np.dtype(float))

    for x_img in range(half_width):
        for y_img in range(image_height):
            if mask[y_img, x_img] == 1:
                x_tilde = x_img / subpixel_nbr
                y_tilde = (y_img - y_offset) / subpixel_nbr
                image[y_img, x_img] = surface_displacement(x_tilde, y_tilde, froude_nbr)

                if progress_bar is not None:
                    progress_bar.update()

    vertical_symetry = np.flip(image, axis=1)
    image = np.concatenate((vertical_symetry, image), axis=1)

    return image


def compute_mask(
    image_height: int,
    image_width: int,
    y_offset: int,
    singularity_offset: float = 0.25,
    blank_mask: bool = False,
    half_mask: bool = True,
    exclude_top_margin: bool = True
):
    """compute mask to avoid computing infinite or zero values"""
    half_length = int(image_width / 2)
    mask = np.ones((image_height, half_length), dtype=bool)

    if not exclude_top_margin:
        y_offset = 0

    if not blank_mask:
        outside_render_angle = 19.47 / 57.4

        for x_mask in range(half_length):
            for y_mask in range(image_height):
                if (x_mask > atan(outside_render_angle)*(y_mask + image_height*singularity_offset) or
                        y_mask < y_offset):
                    mask[y_mask, x_mask] = 0

    if not half_mask:
        vertical_symetry = np.flip(mask, axis=1)
        mask = np.concatenate((vertical_symetry, mask), axis=1)

    return mask
