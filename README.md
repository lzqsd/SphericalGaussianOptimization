# Spherical Gaussian Optimization
This is code to fit per-pixel environment map with spherical Gaussian lobes, using LBFGS optimization. This code has been used in the following paper to generate ground-truth spherical Gaussian parameters.
* Li, Z., Shafiei, M., Ramamoorthi, R., Sunkavalli, K., & Chandraker, M. (2020). Inverse rendering for complex indoor scenes: Shape, spatially-varying lighting and svbrdf from a single image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2475-2484).

Please cite the paper if you find the code to be useful in your own project. Please refer to the papers for more implementation details. 
![](http://cseweb.ucsd.edu/~viscomp/projects/CVPR20InverseIndoor/github/sphericalGaussian.png)

## Instructions
To run the code, use the command
`python optimEnvSplit.py --cuda --dataRoot DATA`, where `DATA` is the path to the synthetic dataset.

