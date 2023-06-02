from math import cos, sin, pi, exp, atan
import tqdm
import numpy as np
import scipy.integrate as integ


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
    # rotate everything by 90°
    x_tilde, y_tilde = y_tilde, x_tilde

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
        progress_bar: tqdm.tqdm = "default",
        relative_vertical_offset: float = 0.05,
        compute_top_margin: bool = False
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
        the right half part of a custom mask as a bool nparray can be
        provided to compute only the interesting areas, by default None.
    progress_bar : tqdm.tqdm, optional
        display a progress bar for the progress of the simulation.It is useful
        as the simulation can take a significant time, by default "default"
        which corresponds to a tqdm.tqdm() instance (search for mutable
        default argument in python for more info).
    relative_vertical_offset : float, optional
        relative space between the top and the singularity, by default 0.05
    compute_top_margin : bool, optional
        whether or not to compute what is in the top margin defined by
        relative_vertical_offset, by default False

    Returns
    -------
    np.ndarray
        2d-matrix representing the z-elevation image
    """
    parametres_combination = (image_height, image_width, max_x_tilde, max_y_tilde)
    if sum([x is None for x in parametres_combination]) != 1:
        raise ValueError("exactly one of the parametres `image_height`, `image_width`,"
                         "`x_tilde_range`, `y_tilde_range` should be None. The others "
                         "must have a value.")

    if image_height is None:
        image_height = image_width / (max_x_tilde * 2) * max_y_tilde

    if image_width is None:
        image_width = image_height / max_y_tilde * (max_x_tilde * 2)

    if max_x_tilde is None:
        max_x_tilde = image_width / image_height * max_y_tilde

    if max_y_tilde is None:
        max_y_tilde = image_height / image_width * (max_x_tilde * 2)

    # to convert image coordinates to simulation coordinates.
    # the image will typically be 400*600 whereas the simulation
    # will be 6 * 10
    subpixel_nbr = image_height / max_y_tilde

    # to move the singularity further down
    y_offset = int(image_height * relative_vertical_offset)

    if mask is None:
        mask = compute_mask(
            image_width,
            image_height,
            mask_y_offset=y_offset,
            blank_mask=(not use_mask),
            compute_top_margin=compute_top_margin
        )

    if progress_bar == "default":
        progress_bar = tqdm.tqdm()

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
    image_width: int,
    image_height: int,
    mask_y_offset: int,
    cone_y_offset: float = 0.25,
    outside_render_angle: float = 19.47,
    blank_mask: bool = False,
    compute_top_margin: bool = False
) -> np.ndarray:
    """
    Compute the right half of a mask to avoid computing integral that
    are going to be null anyway. This is based on the Kelvin angle
    (19.47°) for a wake pattern.

    Parameters
    ----------
    image_height : int
        the image height in pixel.
    image_width : int
        the image width in pixel.
    mask_y_offset : int
        shift the mask vertically. Useful when there is an offset to
        the simulation.
    cone_y_offset : float, optional
        relative shift for the cone. It appears that the vertex of the
        cone drawn by the wake is higher than the top of the frame
        for small froude number, so this parameter compensate for that,
        by default 0.25
    outside_render_angle : float, optional
        angle (in degree) above which the surface elevation is not
        computed. This spare a lot of computational ressource and avoids
        integral approximation, by default 19.47° (Kelvin angle)
    blank_mask : bool, optional
        to generate a mask full of ones, by default False
    compute_top_margin : bool, optional
        avoids the exclusion of the top margin from the computation,
        by default False

    Returns
    -------
    np.ndarray
        boolean 2d-array mask
    """
    half_length = int(image_width / 2)
    mask = np.ones((image_height, half_length), dtype=bool)

    if compute_top_margin:
        mask_y_offset = 0

    if not blank_mask:
        outside_render_angle = outside_render_angle / 57.4

        for x_mask in range(half_length):
            for y_mask in range(image_height):
                offset_y_mask = y_mask + image_height * cone_y_offset
                # check if current coord are outside the render cone or
                # above the mask_y_offset
                if (x_mask > atan(outside_render_angle) * offset_y_mask or
                        y_mask < mask_y_offset):
                    mask[y_mask, x_mask] = 0

    return mask


def elevation_vs_angle(
    simulation_result: np.ndarray,
    relative_vertical_offset: float,
    n_angles_samples: int = 100,
    max_angle: float = 50,
    min_relative_distance: float = 0.35
) -> np.ndarray:
    """
    outputs the maximum relative elevation along multiple lines forming
    different angles with the vertical. Corresponds to Z(phi) / Zmax,
    with Zmax being the maximum of the simulation result.

    Parameters
    ----------
    simulation_result : np.ndarray
        2D-array of the simulation.
    relative_vertical_offset : float
        vertical origin of the perturbation.
    n_angles_samples : int, optional
        number of lines to draw, by default 10.
    max_angle : float, optional
        maximum angle to go to, by default 50.
    min_relative_distance : float, optional
        exclude points that are too close to the pertubation to avoid
        messing the plot, by default 0.35

    Returns
    -------
    np.ndarray
        1D-array of relative maximum elevation versus phi.

    Example
    -------
    ```python
    from simulation import elevation_vs_angle, linspace

    froude_10_results = np.load(Path("froude=1.0.npy"))
    relative_elevations = elevation_vs_angle(froude_10_results, relative_vertical_offset=0.05, min_relative_distance=0.2)

    plt.plot(list(linspace(0, 50, 100)), relative_elevations)
    plt.ylabel("Relative elevation")
    plt.xlabel("phi")

    plt.axvline(x = 19.47, linestyle=":", label="Kelvin angle")

    plt.title("Relative elevation for a froude number of 1.0")
    plt.show()
    ```
    """
    image_height, image_width = simulation_result.shape[:2]

    elevation = np.zeros(n_angles_samples)

    for i, phi in enumerate(linspace(0, max_angle, n_angles_samples)):
        line_elevation = []

        generator = polar_image_coord(
            phi, image_width, image_height,
            int(image_width / 2), int(image_height * relative_vertical_offset)
        )
        for _, x_img, y_img in generator:
            if y_img > min_relative_distance * image_height:
                z = simulation_result[y_img, x_img]
                line_elevation.append(abs(z))

        if len(line_elevation) == 0:
            value_elevation = 0
        else:
            value_elevation = np.percentile(line_elevation, 95)
        elevation[i] = value_elevation

    elevation /= np.max(elevation)

    return elevation


def draw_diagonals(
    simulation_result: np.ndarray,
    relative_vertical_offset: float,
    nbr_of_lines: int = 10,
    max_angle: float = 50,
    min_relative_distance: float = 0.35
) -> np.ndarray:
    """
    draw the diagonal lines along which are computed the "elevation vs
    angle".

    Parameters
    ----------
    simulation_result : np.ndarray
        2D-array of the simulation.
    relative_vertical_offset : float
        vertical origin of the perturbation.
    nbr_of_lines : int, optional
        number of lines to draw, by default 10.
    max_angle : float, optional
        maximum angle to go to, by default 50.
    min_relative_distance : float, optional
        exclude points that are too close to the pertubation to avoid
        messing the plot, by default 0.35

    Returns
    -------
    np.ndarray
        2D-array of the simulation with lines drawn onto it.
    """
    image_height, image_width = simulation_result.shape[:2]

    max_value = np.max(simulation_result)

    for phi in linspace(0, max_angle, nbr_of_lines):
        generator = polar_image_coord(
            phi, image_width, image_height,
            int(image_width / 2), int(image_height * relative_vertical_offset)
        )
        for _, x_img, y_img in generator:
            if y_img > min_relative_distance * image_height:
                simulation_result[y_img, x_img] = 1.3 * max_value

    return simulation_result


def linspace(start: float, stop: float, n: int):
    """linspace generator"""
    if n == 1:
        yield stop
        return
    h = (stop - start) / (n - 1)
    for i in range(n):
        yield start + h * i


def pol2cart(rho: float, phi: float) -> tuple[int, int]:
    """
    convert polar coordinates to cartesians.

    Parameters
    ----------
    rho : float
        distance
    phi : float
        angle

    Returns
    -------
    tuple[int, int]
        the x and y coordinates corresponding to the polar coordinates.
    """
    x = int(rho * np.cos(phi))
    y = int(rho * np.sin(phi))
    return (x, y)


def polar_image_coord(
    phi: float,
    image_width: int,
    image_height: int,
    x_origin: int,
    y_origin: int
):
    """
    generate the x and y coordinate along a line given its angle.

    Parameters
    ----------
    phi : float
        angle to the vertical, in degree.
    image_width : int
        the width of the image.
    image_height : int
        the height of the image.
    x_origin : int
        the x origin of the line.
    y_origin : int
        the y origin of the line.

    Yields
    ------
    tuples[float, int, int]
        return the value of rho, x, and y
    """
    x = 0
    y = 0
    prev_x = -1
    prev_y = -1

    rho = 0

    while True:
        x, y = pol2cart(rho, np.pi / 2 - phi / 57.4)

        if (x + x_origin) >= image_width or (y + y_origin) >= image_height:
            break

        if x != prev_x or y != prev_y:
            corrected_x = x + x_origin
            corrected_y = y + y_origin

            prev_x = x
            prev_y = y

            yield rho, corrected_x, corrected_y

        rho += 1
