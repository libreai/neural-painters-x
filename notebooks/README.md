# Joy of Neural Painting Notebooks

This folder contains our implementation for **Neural Painters** [1].

## Notebook Descriptions

1. [01_neural_painter_x_training_Generator_Non_Adversarial.ipynb](https://github.com/libreai/neural-painter-x/blob/master/notebooks/01_neural_painter_x_training_Generator_Non_Adversarial.ipynb) Pre-trains the Generator with a non-adversarial loss, e.g., using a feature loss (also known as perceptual loss).

2. [02_neural_painter_x_training_Critic_Non_Adversarial.ipynb](https://github.com/libreai/neural-painters-x/blob/master/notebooks/02_neural_painter_x_training_Critic_Non_Adversarial.ipynb) Pre-trains the Critic as a Binary Classifier
(i.e., non-adversarially) using the pre-trained Generator (in evaluation mode with frozen model weights) to generate `fake` brushstrokes. That is, the Critic should learn to discriminate between real images and the generated ones. This step uses a standard binary classification loss, i.e., Binary Cross Entropy, not a GAN loss.

3. [03_neural_painter_x_training_GAN_mode.ipynb](https://github.com/libreai/neural-painters-x/blob/master/notebooks/03_neural_painter_x_training_GAN_mode.ipynb) continues the Generator and Critic training in a GAN setting. Faster!

4. [04_neural_painter_x_painting.ipynb](https://github.com/libreai/neural-painters-x/blob/master/notebooks/04_neural_painter_x_painting.ipynb) implements the Intrinsic Style Transfer approach introduced in [1] using a the Feature Loss introduced in [Fast.ai's MOOC](https://course.fast.ai/videos/?lesson=7).

### References

[1] Neural Painters: A Learned Differentiable Constraint for Generating Brushstroke Paintings. Reiichiro Nakano 
arXiv preprint [arXiv:1904.08410]((https://arxiv.org/abs/1904.08410)), 2019. [Github repo](https://github.com/reiinakano/neural-painters).
