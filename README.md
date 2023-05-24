# Wake pattern simulation

All the physics formula in this code come from the paper [*Kelvin wake pattern at large Froude numbers*](https://www.pct.espci.fr/~elie/Publications/Kelvin_wake_pattern.pdf)

## Wake simulation

```python
def wake_simulation(...) -> np.ndarray:
```

You can simulate the wake pattern via the function `wake_simulation`. It can take a long time on a decent CPU if you want to generate high definition images or if you are computing large froude number (> 1).

Parameters

- froude_nbr : float
    - froude number for the simulation
- image_height : int, optional
    - the height of the output image. If set to None, will be computed
    using `image_width`, `max_x_tilde`, `max_y_tilde`, by default 600
- image_width : int, optional
    - the width of the output image. If set to None, will be computed
    using `image_height`, `max_x_tilde`, `max_y_tilde`, by default None
- max_x_tilde : float, optional
    - the max X coordinate for the simulation. If set to None, will be
    computed using `image_width`, `image_height`, `max_y_tilde`, by default 3
- max_y_tilde : float, optional
    - the max X coordinate for the simulation. If set to None, will be
    computed using `image_width`, `image_height`, `max_x_tilde`, by default 10
- use_mask : bool, optional
    - whether to use a mask to avoid computing null area. Using a mask
    for low froude number can truncate non-null areas, by default True
- mask : np.ndarray, optional
    - the right half part of a custom mask as a bool nparray can be
    provided to compute only the interesting areas, by default None.
- progress_bar : tqdm.tqdm, optional
    - display a progress bar for the progress of the simulation.It is useful
    as the simulation can take a significant time, by default tqdm.tqdm()
- relative_vertical_offset : float, optional
    - relative space between the top and the singularity, by default 0.05
- compute_top_margin : bool, optional
    - whether or not to compute what is in the top margin defined by
    relative_vertical_offset, by default False

Returns

np.ndarray
    2d-matrix representing the z-elevation image

### Example

![Wake pattern simulations](ressources/Wake%20pattern%20simulations.jpg)

The individual images has been generated using the following code:

```python
from simulation import wake_simulation
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

simulation_fr_10 = wake_simulation(1.0)
# as it takes a significant time to generate the image, we save it
np.save("results/froude=1.0.npy", simulation_fr_10)

# this ensure that the background on each simulation is the same color
normalized_image = (simulation_fr_10 - np.min(simulation_fr_10)) / (np.max(simulation_fr_10) - np.min(simulation_fr_10)) * 2

plt.figure(figsize=(10, 14))

colour_offset = 0.7
plt.imshow(simulation_fr_10 - colour_offset, cmap="twilight", norm=colors.CenteredNorm(0))

plt.axis('off')
plt.tight_layout()
plt.show()
```

## Relative elevation versus angle

outputs the maximum relative elevation along multiple lines forming different angles with the vertical. Corresponds to Z(phi) / Zmax, with Zmax being the maximum of the simulation result.

```python
def elevation_vs_angle(...) -> np.ndarray:
```

Parameters

- simulation_result : np.ndarray
    - 2D-array of the simulation.
- relative_vertical_offset : float
    - vertical origin of the perturbation.
- n_angles_samples : int, optional
    - number of lines to draw, by default 10.
- max_angle : float, optional
    - maximum angle to go to, by default 50.
- min_relative_distance : float, optional
    - exclude points that are too close to the pertubation to avoid
    messing the plot, by default 0.35

Returns

np.ndarray
    1D-array of relative maximum elevation versus phi.

internally, it computes the 95th percentile (to avoid integral estimation errors) along multiple lines as shown in the next picture. The lines do not start at the perturbation for reasons mentionned in the paper.

![Relative elevation computation method](ressources/Relative%20elevation%20computation%20method.png)

### Example

![Relative elevation for a froude number of 1.0](ressources/Relative%20elevation%20for%20a%20froude%20number%20of%201.0.png)

```python
from simulation import elevation_vs_angle, linspace

simulation_fr_10 = np.load(Path("results/froude=1.0.npy"))

relative_elevations = elevation_vs_angle(simulation_fr_10, relative_vertical_offset=0.05, min_relative_distance=0.2)

plt.plot(list(linspace(0, 50, 100)), relative_elevations)
plt.ylabel("Relative elevation")
plt.xlabel("phi")

plt.axvline(x = 19.47, linestyle=":", label="Kelvin angle")
plt.title("Relative elevation for a froude number of 1.0")

plt.show()
```
