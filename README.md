# linear_segmentation
Automatic sparse sampling of 1-D array into linear segments minimizing error

# Install
pip install .

# Example
python -m linear_segmentation.examples.linear_segmentation_with_slider.py

# Usage
```
from linear_segmentation import linear_segmentation
import scipy.interpolation as spint

...
x, y = linear_segmentation(data, tol=0.001)  # normalized abs error tolerance
data_interp = spint.interp1d(x, y)
...
new_y = data_interp(new_x)
