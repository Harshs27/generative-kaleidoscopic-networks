# Generative Kaleidoscopic Networks
`Dataset Kaleidoscope`: We discovered that the Deep ReLU networks (or Multilayer Perceptron architecture) demonstrate an 'over-generalization' phenomenon. In other words, the MLP learns a many-to-one mapping and this effect is more prominent as we increase the number of layers or depth of the MLP. We utilize this property of Deep ReLU networks to design a dataset kaleidoscope, termed as 'Generative Kaleidoscopic Networks'. 

`Kaleidoscopic Sampling`: If we learn a MLP to map from input $x\in\mathbb{R}^D$ to itself $f_\mathcal{N}(x)\rightarrow x$, the sampling procedure starts with a random input noise $z\in\mathbb{R}^D$ and recursively applies $f_\mathcal{N}(\cdots f_\mathcal{N}(z)\cdots )$. After a burn-in period duration, we start observing samples from the input distribution and we found that deeper the MLP, higher is the quality of samples recovered. 

`Research`: Generative Kaleidoscopic Networks (on [arxiv](<https://arxiv.org/abs/xxx.xxx>))

## The 'Over-generalization' Phenomenon in Neural Networks
`Over-generalization`: If the output units of neural networks are bounded, they tend to 'over-generalize' on the input data. That is, the output values over the entire input range are mapped close to the output range that were seen during learning, exhibiting a `many-to-one' mapping.

- (Analysing **1D** space) Manifold learning is supposed to reconstruct the input at the output. Manifold learning was done on data points $X=(0.2, 0.8)$ by fitting a MLP with $H=5,L=7$, intermediate layers had 'ReLU' non-linearity and 'Sigmoid' at the final layer. We observe the flat regions around the points $(0.2, 0.8)$ in the output manifold where the MLP learns many-to-one mapping, thus indicates the existence of the over-generalization phenomenon. The 'red' points indicate the initial noise, while the 'green' points indicate the intermediate instances of the kaleidoscopic sampling procedure.
<p align="center">
<img src="https://github.com/Harshs27/generative-kaleidoscopic-networks/blob/main/saved_samples/synthetic_1D_manifold_profiles.png" width="150" alt="synthetic_1D_manifold_profiles"/>    <img src="https://github.com/Harshs27/generative-kaleidoscopic-networks/blob/main/saved_samples/synthetic_1D_run20.gif" width="300" alt="synthetic_1D_kaleidoscopic_samples"/>
</p>

- (Analysing **2D** space) Manifold learning in 2-dimensions with a MLP $L=7, H=50$ at the points $x=[(0.2,0.2), (0.2,0.8), (0.8,0.2), (0.8,0.8)]$. 'Red' points are the initial random noise sampled from a Normal distribution $\mathcal{N}(0.5, 0.5I)$. The convergence of the Kaleidoscopic sampling procedure are shown with samples in 'green'. The loss function hyperplane is quite flat but still a large number of samples are obtained near the input distribution which indicates that our sampling procedure is working which in turn suggests the existence of the `over-generalization' phenomenon.
<p align="center">
<img src="https://github.com/Harshs27/generative-kaleidoscopic-networks/blob/main/saved_samples/synthetic_2D_run15.gif" width="400" alt="synthetic_2D_kaleidoscopic_samples"/>
</p>

## The MNIST Kaleidoscope

- (Creating `MNIST Kaleidoscope`) Manifold learning is done on the MNIST data images with a MLP of $L=10, H=500$, intermediate 'ReLU' and final layer with 'Tanh' non-linearity. We start with input noise vector sampled from a Uniform distribution $~\mathcal{U}(-1,1)$. The images show the consecutive runs after applying learned MLP on the input noise. We note that after a certain 'burn-in' period, the procedure can converge at a digit and then it can remain stable throughout the future iterations as it has found a stable minima or step as defined in our analysis. For this reason, we add a slight noise, parameter $\epsilon=0.01$, at every sampling iteration to simulate a kaleidoscopic effect. The gif shows the output of randomly chosen subset of images and the seed is fixed throughout, so that there is one-to-one correspondence across sampling runs. One can take a digit and observe their evolution over the sampling iterations. 
<p align="center">
<img src="https://github.com/Harshs27/generative-kaleidoscopic-networks/blob/main/saved_samples/mnist_kaleidoscope.gif" width="400" alt="mnist_kaleidoscopic_samples"/>
</p>

## Setup  
The `setup.sh` file contains the conda environment details. run `bash setup.sh`    
In case of dependencies conflict, one can alternatively use this command `conda env create --name kals --file=environment.yml`.  

## Citation
If you find this method useful, kindly cite the associated paper:

@article{
}   

## Contact
Email: harshshrivastava111@gmail.com
