# Matrixopt
Library that simplifies matrix optics calculations.

## Key features
  - Provides plenty of optical elements
  - Simulates propagating through the optical system and prints out the resultant gaussian beam

# TO-DO
  - Add optimization module to iterate though possible optical system compositions and choose the most favourable one 
  - Support non-Gaussian beams
  - Prints out the scheme of the system
  - Prints out the gaussian beam transformation
  - ... ?

## Usage
```Python
from matrixopt.ABCDformalism *
input = GaussianBeam(wavelength=405e-9, zr=0.01)

fs1 = FreeSpace(0.1)
tl1 = ThinLens(2.5e-2)
fs2 = FreeSpace(0.2)
tl2 = ThickLense(0.8, 1.2, 0.4, 0.01)
fs3 = FreeSpace(1)

op = OpticalPath(fs1, tl1, fs2, tl2, fs3)
output = op.propagate(input)
print(output)
```
