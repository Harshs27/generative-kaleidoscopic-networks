# Generative Kaleidoscopic Networks
`Dataset Kaleidoscope`: We discovered that the Deep ReLU networks (or Multilayer Perceptron architecture) demonstrate an 'over-generalization' phenomenon. In other words, the MLP learns a many-to-one mapping and this effect is more prominent as we increase the number of layers or depth of the MLP. We utilize this property of Deep ReLU networks to design a dataset kaleidoscope, termed as 'Generative Kaleidoscopic Networks'. 

`Kaleidoscopic Sampling`: If we learn a MLP to map from input $x\in\mathbb{R}^D$ to itself $f_\mathcal{N}(x)\rightarrow x$, the sampling procedure starts with a random input noise $z\in\mathbb{R}^D$ and recursively applies $f_\mathcal{N}(\cdots f_\mathcal{N}(z)\cdots )$. After a burn-in period duration, we start observing samples from the input distribution and we found that deeper the MLP, higher is the quality of samples recovered. 

`Research`: Generative Kaleidoscopic Networks [arxiv](<https://arxiv.org/abs/xxx.xxx>)  

## The 'Over-generalization' Phenomenon
`Over-generalization`: If the output units of neural networks are bounded, they tend to 'over-generalize' on the input data. That is, the output values over the entire input range are mapped close to the output range that were seen during learning, exhibiting a `many-to-one' mapping.

- (Analysing **1D** space) Manifold Learning done on points [0.2, 0.8]. We run manifold learning on data points $X=\{0.2, 0.8\}$ by fitting a MLP with $H=5,L=2$ (top 2 rows) and $H=5,L=7$ (bottom 2 rows) and in each case, we run for epochs $\geq 1K$ to ensure that the training loss tends to $\rightarrow 0$. Intermediate layers had `ReLU' non-linearity and `Sigmoid' at the final layer. By design, the manifold learning is supposed to match the input and output. But, on the contrary, we can observe the over-generalization phenomenon in (b,h), where the MLP learns many-to-one mapping, as evident by the flat regions around the points $\{0.2, 0.8\}$. Scaled version of loss shown in (a,g) on the zoomed in part as the increasing and spreading the training points across the space makes the loss manifold flatter. The rows (c-f) and (i-l) shows intermediate instances of the sampling runs. The burn-in period is roughly around $B=5$, which indicates that the sampling converges fast.

<p align="center">
<img src="https://github.com/Harshs27/generative-kaleidoscopic-networks/blob/main/saved_samples/synthetic_1D_manifold_profiles.png" width="150" alt="synthetic_1D_manifold_profiles"/>    <img src="https://github.com/Harshs27/generative-kaleidoscopic-networks/blob/main/saved_samples/synthetic_1D_run20.gif" width="300" alt="synthetic_1D_kaleidoscopic_samples"/>
</p>

- (Analysing **2D** space) Manifold Learning done on [(0.2, 0.2), (0.2, 0.8), (0.8, 0.2), (0.8, 0.8)]. We do manifold learning in 2-dimensions with a MLP $L=2, H=50$ at point $x=(0.5,0.5)$ and (b) a MLP $L=7, H=50$ at the points $x=[(0.2,0.2), (0.2,0.8), (0.8,0.2), (0.8,0.8)]$. Each plot initializes a random noise (in {\color{bittersweet} red}) sampled from a Normal distribution $\mathcal{N}(0.5, 0.5I)$ and ran \kals sampling whose samples are shown in {\color{aoenglish}green}. The loss function hyperplane is quite flat but still a large number of samples are obtained near the input distribution which indicates that our sampling procedure is working which in turn suggests the existence of the `over-generalization' phenomenon.
  
<p align="center">
<img src="https://github.com/Harshs27/generative-kaleidoscopic-networks/blob/main/saved_samples/synthetic_2D_run15.gif" width="400" alt="synthetic_2D_kaleidoscopic_samples"/>
</p>

## The MNIST Kaleidoscope

- (Creating `MNIST Kaleidoscope`) We do manifold learning on a MLP with $L=10, H=500$ with intermediate 'ReLU' and final layer with 'Tanh' non-linearity. We create a 'MNIST Kaleidoscope' by setting $\epsilon=0.01$ in the 'Kaleidoscopic' sampling procedure. [TOP row] The leftmost images show the digits recovered after the manifold learning. The center images show the $1^{st}$ run after applying learned MLP on the input noise. The input noise vector was sampled randomly from a Uniform distribution $~\mathcal{U}(-1,1)$. (Note, we get similar results with Normal distribution too). The rightmost images show the state at the sampling run of 300. We note that the procedure can converge at a digit and then it can remain stable throughout the future iterations as it has found a stable minima or step as defined in our analysis. For this reason, we add a slight noise at every sampling iteration to simulate kaleidoscopic behaviour. Here, we show the output of randomly chosen subset of images and the seed is constant, so that there is one-to-one correspondence across sampling runs. [MIDDLE row] A sequence of intermediate sampling runs $5\rightarrow13$, which is still in the burn-in period. One can take a digit and observe their evolution over the iterations. [TODO: Explain the one-to-one correspondence over the sampling runs.]

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
