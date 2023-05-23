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


def surface_displacement(coordinates: tuple[float, float], froude_nbr: float) -> float:
    x_tilde, y_tilde = coordinates

    value, _ = integ.quad(fonction_a_integrer,
                          a=-pi / 2,
                          b=pi / 2,
                          args=(x_tilde, y_tilde, froude_nbr),
                          limit=250,
                          epsabs=1e-4,
                          epsrel=1e-4,)
    return -value


def image_simulation(froude_nbr: float,
                     use_mask: bool = True,
                     progress_bar: tqdm.tqdm = None,
                     exclude_top_margin: bool = True):
    précision = résolution_Y/(interval_Y[1]-interval_Y[0])
    half_length = int(résolution_X / 2)

    # to move the singularity further down
    y_offset = int(résolution_Y * 0.05)

    mask, pixel_count = compute_mask((résolution_X, résolution_Y),
                                     y_offset=y_offset,
                                     blank_mask=(not use_mask),
                                     exclude_top_margin=exclude_top_margin,)

    image = np.zeros((résolution_Y, half_length), dtype=np.dtype(float))

    progress_bar = tqdm.tqdm(total=pixel_count)

    for x_img in range(half_length):
        for y_img in range(résolution_Y):
            if mask[y_img, x_img] == 1:
                x_simu = x_img / précision
                y_simu = (y_img - y_offset) / précision
                z_elevation = surface_displacement((y_simu, x_simu), froude_nbr)

                image[y_img, x_img] = z_elevation
                progress_bar.update()

    vertical_symetry = np.flip(image, axis=1)
    image = np.concatenate((vertical_symetry, image), axis=1)

    return image


def compute_mask(size: tuple[int, int],
                 y_offset: int,
                 blank_mask: bool = False,
                 half_mask: bool = True,
                 exclude_top_margin: bool = True):
    """compute mask to avoid computing infinite or zero values"""
    half_length = int(size[0]/2)
    mask = np.ones((size[1], half_length), dtype=bool)

    pixel_count = size[1] * half_length

    if not exclude_top_margin:
        y_offset = 0

    if not blank_mask:
        outside_render_angle = 19.57 / 57.4

        for x_mask in range(half_length):
            for y_mask in range(size[1]):
                if (x_mask > atan(outside_render_angle)*(y_mask + size[1]*0.25) or
                        y_mask < y_offset):
                    mask[y_mask, x_mask] = 0
                    pixel_count -= 1

    if not half_mask:
        vertical_symetry = np.flip(mask, axis=1)
        mask = np.concatenate((vertical_symetry, mask), axis=1)

    return mask, pixel_count
