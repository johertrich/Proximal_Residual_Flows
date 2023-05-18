# Proximal Residual Flows for Bayesian Inverse Problems

This code belongs to the paper [4]. Please cite the corresponding paper, if you use this code.

The paper is available at  
https://doi.org/10.1007/978-3-031-31975-4_16  
An Arxiv preprint is available at
https://arxiv.org/abs/2211.17158

The repository contains an implementation of Proximal Residual Flows as introduced in [4]. 
Proximal Residual Flows are an architecture for normalizing flows based on invertible residual flows [1, 2] and proximal neural networks [3, 5].
The repository contains the following scripts for reproducing the numerical examples in [4].  

- `toy_examples.py` for the toy density examples from Section 4.1.

- `circle_example.py` for the circle example from Section 4.2.

- `mixture_models.py` for the mixture example from Section 4.2.

More examples will be added soon.

The code is written in python with Tensorflow 2.10.0.

For questions, bugs or any other comments, please contact Johannes Hertrich (j.hertrich(at)math.tu-berlin.de).

## REFERENCES

[1] J. Behrmann, W. Grathwohl, R. T. Chen, D. Duvenaud, and J.-H. Jacobsen.  
Invertible residual networks.  
In International Conference on Machine Learning, pages 573–582, 2019.

[2] R. T. Q. Chen, J. Behrmann, D. K. Duvenaud, and J.-H. Jacobsen.  
Residual flows for invertible generative modeling.  
In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

[3]  M. Hasannasab, J. Hertrich, S. Neumayer, G. Plonka, S. Setzer, and G. Steidl.  
Parseval proximal neural networks.  
Journal of Fourier Analysis and Applications, 26:59, 2020.

[4] J. Hertrich.  
Proximal Residual Flows for Bayesian Inverse Problems.  
L. Calatroni, M. Donatelli, S. Morigi, M. Prato and M. Santacesaria (eds.)  
Scale Space and Variational Methods in Computer Vision.  
Lecture Notes in Computer Science, 14009, 210-222, 2023.  

[5] J. Hertrich, S. Neumayer and G. Steidl.  
Convolutional Proximal Neural Networks and Plug-and-Play Algorithms.  
Linear Algebra and its Applications, vol 631 pp. 203-234, 2021.



