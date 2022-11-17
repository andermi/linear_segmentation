# linear_segmentation
Automatic sparse sampling of 1-D array into linear segments minimizing error

# Install from clone
`pip install .`

# Install from PYPI
`pip install linear_segmentation`

# Example
`python -m linear_segmentation.examples.linear_segmentation_with_slider.py`

# Usage
```
from linear_segmentation import linear_segmentation
from linear_segmentation import refinement  # optional
import scipy.interpolation as spint

...
x, y = linear_segmentation(data, tol=0.001)  # normalized abs error tolerance
x_refined, y_refined = refinement(data, x)  # optional refinement using least squares
data_interp = spint.interp1d(x_refined, y_refined)  # or x, y
...
new_y = data_interp(new_x)
```
