"""
x = crystal structure, model parameters i.e. varibles
y = f(x)
y = Discreate (classification) or continous (regression) i.e. the prediction

First question -> what is the data?
for a perovskite for example. is it a crystal, ratios, ect? what is important

take vector with a fixed length and import the vector into the model.

If the data can be modeled the function doesn't really matter. However, the success all depends on the data and the structure of the data.

Sources of data:
1) High throughput data
2) Data mining and scrape papers
3) Theoretical

Experimential science vs cats (image recognition)
Varible    Images  Molecules/materials
Quantity    1m  1000
Noise   Low (labels normally correct)   Depending on the source
Ambiguity   Low High(different way to measure property)
Diversion   Low High

ML interpolate but cannot extrapolate. Higher the diversity better the prediction
DFT should be validated in a diverse material set. Not always true

How to construct good features? Four ways
1) Computability - generating the feature must be less computatiavly expensive than DFT (or other) computations or experiment.
2) Uniqueness - one feature vector or one material
3) Simularity - simular materials should have simular features
4) Symmetry - must respect all physics

These questions should be asked when looking at papers.

##### Feature generation
How to construct good features?
i.e. Goldschmidt 1926 geometry approach
fmats.2016.00019 - https://www.frontiersin.org/articles/10.3389/fmats.2016.00019/full

How about a brute force search all the features of the crystal structure
Create huge d{} of all descriptors.
Which combination of descriptors are best to describe property x
Physical review letter 2016 16028

Can we construcutre more blackbox features
MAGPIE = materials agnostic platform for informatics and exploration
Magpie a 145 dimentional vector for features.
Magpie feature take vector -> take differnt lp norms
take all the properties and then take max, min, range, std, mode ect.
Using these values is it possible to make a prediction. Weak correlators used to predict properties.
npj computational materials 29 (2018) - used random forest.
https://www.nature.com/articles/s41524-018-0085-8
Correlation doesn't mean that a feature is the causation.

##### Model infence
1) Linear reg
y=bx y=pred, beta=weight, x=property
Make an assumption that that noise (e) is independent of property x
y=bx + e
This is not always true for material science (homoscedasticity)
This can be solved analytically
Note that n < p (number of data points less than the number of inputs)
(Ridge regression)
Must check the data set with cross validation. General used test_train_split and the run a cross validation after. This can be easily implimated on skikit learn.
This is the overfitting and underfitting problem.
Known as the bias variance tradeoff.
Three contributions to erros
1) noisy data high error 2) how senstitive is the model to data variability 3) how close is the model to the ground truth.
Generally train, test, crossvalidate, tweak parameters and then train, test, has the cross validation improved?
Common priors in linear regression -> beta is space (look up)
Example PNAS 113, 3932 (2016):
Four reagents (A,B,C,D) write down rate constant equation
Lasso regression compared with sparce and non space (line 71)

2) Deep learning
Stacked linear regression.
Y=out(W_n\sigma(W_N-1\sigma(...W_1\X)))
out(.) is the output layer and sigma is known as the activation function (type of function doesn't really matter but must be non-linear)
width is y and x is depth or NN
Chemistry_through_transfer_leanring_674440
https://chemrxiv.org/articles/Outsmarting_Quantum_Chemistry_Through_Transfer_Learning/6744440
Issues: many many hyperparameters, architecture (number of layers, number of neurons per layer) +2 more didn't write down.
How to include sysmmetry in the models. Translation and permutatioanl sym? via convolution idea
Shift output and input via a convolution
https://en.wikipedia.org/wiki/Convolutional_neural_network
Enforcing equivarience through message passing
https://arxiv.org/abs/1704.01212
Graph convolution given as an exampe PRL, 120, 145301 (2018)
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301

3) Gaussian process
Can gain uncertainity and predictions
https://en.wikipedia.org/wiki/Gaussian_process
https://pubs.acs.org/doi/abs/10.1021/acsnano.8b04726
bayesian opermation to neural networks hyperparameters
Never using  a grid search.
PCA on final layers of last hidden layer in NN
Low dimensional repersenation ..?


Challange
Question: Predict experimentally measured band gap of materials
Stoichiometric formular


"""
