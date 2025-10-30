This work introduces a framework for optimizing deep ensmebles using Bayesian optimization, and Sobol sequence to build the prior distribution. The approach is scalable and could be applied to any type of neural networks. The current implementation include a test case on a Densely Connected Convolution Neural Network (DCNN) and a Multi-Layer Perception (MLP) neural network. 
The DCNN is trained on computational data, and noise is introduced to the data to asses the method performance on noisy data. The MLP is trained on experimental data.  

Problem statement:
When DEs are not adequately optimized or constructed using a fixed DNN architecture and only varied through random weight and bias initializations, their performance may degrade for highly nonlinear or high-dimensional problems.
To enhance the performance of DEs, different post-hoc calibration methods, are applied after the ensmble is trained, are suggested such as temperature scaling, isotonic regression. This approach despite it improves
the prediction, it wouldn’t improve the internal predictive structure of the models and thus wouldn’t lead to best UQ and accuracy. Recent research has emphasized the integration of optimization techniques to systematically
enhance both predictive accuracy and uncertainty quantification performance. Optimizing multiple deep neural network is expensive, that's why a framework for conducting the optimization is needed. 

We adopt several efficiency-enhancing strategies: utilizing a Sobol sequence to initialize the BO search space, leveraging parallel computing to optimize each ensemble member on a separate processor, 
conducting optimization on a representative smaller dataset, reducing hyperparameter dimensionality, constraining hyperparameter bounds based on prior knowledge, fixing the number of neural network training iterations, 
and limiting the number of optimization iterations.

Methodology:
<img src="Figures/optimization_strategy.jpg" width="60%" alt="Pipeline">
