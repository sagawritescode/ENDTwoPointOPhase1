# What is a neural network neuron?

A neural network neuron is a node in the layer. It's name is inspired from the biological neurons in brain. It acts as a intermediatary memory storage and the value it holds is later transformed using weights and activation function. 

# What is the use of the learning rate?

The weights are corrected using the error computed from the result on the training data. The initial/previous weights are substracted with the product of learning rate and the partial derivative of the error w.r.t to that weight. The learning rate acts as a multiplicative factor for the partial derivative, hence it affects the value of the newer weight after each epoch. Smaller learning rate will take comparatively more number of epochs to get the desired weights while larger rate might also delay the training as it will miss the minimum (of the error curve). Optimal learning rate reaches the minima point in reasonable number of epochs

# How are weights initialized?

In this assignment (code part), the weights are initialized using normal distribution

# What is "loss" in a neural network?

Loss is the difference between the predicted output and the actual output. While training, loss is used to update and train the weights

# What is the "chain rule" in gradient flow?

For training/updating the weights we subtract the previous weight value with the partial derivative of the error w.r.t the weight. But this partial derivative is not directly known. The chain rule of the gradient flow can be used to link known derivatives and derive this value. For example, lets say there are n layers in a neural network and w211 (weight connecting first neuron in first layer to first neuron in second layer) is to be trained. But partial derivative of error w.r.t w211 is not known. But again, partial derivative w.r.t last layer (n) is known. Using the chain rule in calculus, we can compute the partial derivative of error w.r.t to any weight


