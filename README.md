# FSGMTutorial
The original paper can be found at:
https://arxiv.org/abs/1412.6572

We begin with deriving a simple way of constructing an adversarial example around an input (x, y).
Supppose we denote our neural network by a function f : X → {0, . . . , 9}.
Suppose we want to find a small perturbation ∆ of x such that the neural network f assigns a label
different from y to x+∆. To find such a ∆, we want to increase the cross-entropy loss of the network f
at (x, y); in other words, we want to take a small step ∆ along which the cross-entropy loss increases,
thus causing a misclassification. We can write this as a gradient ascent update, and to ensure that we
only take a small step, we can just use the sign of each coordinate of the gradient. The final algorithm
is this:
x˜ = x + ε · sign(∇L(f, x, y)),
where L is the cross-entropy loss.

First, implement the Fast Gradient Sign Method (FGSM) for the neural network given to you in the last tutorial. Then,
evaluate and report the accuracy of the neural network on adversarial examples. This is computed
as follows – 
for each test example x(i)
, generate an adversarial example ˜x(i)
for ε= 0.1. 
The neural
network is correct if it predicts y(i) on ˜x(i) and wrong otherwise.

