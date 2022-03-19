## Optimization module
### TO-DO:
Desired behaviour is
```python
from optix.matrixopt import *
op = Optimizer()
op.append(FreeSpace)                        # Unspecified propagation length
op.append(PlanoConvexLens(10e-3,2e-2,1.5))  # Fixed lens planoconvex lens
op.append(FreeSpace(10e-2))                 # Fixed propagation length
op.append(ThinLens)                         # Unspecified lensed focal length

d, f = op.run(GAUSS_IN, 
              desire=(W0_OPTIMUM, W0_LOC_OPTIMUM), 
              tolerance=(1e-3, 10e-2),
              x0=(10e-2, 400e-3))
```