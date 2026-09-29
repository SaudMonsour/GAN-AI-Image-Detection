# Sources and attribution

- Bird, J. J. and Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access. https://arxiv.org/abs/2303.14126
- Author dataset repository: https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images/tree/e112a942abaecd02b6b1f6f646c807d56be8fb62
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- Goodfellow et al. (2014). Generative Adversarial Nets. https://arxiv.org/abs/1406.2661
- Radford, Metz, and Chintala (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. https://arxiv.org/abs/1511.06434
- TensorFlow DCGAN tutorial: https://www.tensorflow.org/tutorials/generative/dcgan

CIFAKE's authors distribute the dataset under an MIT license statement, retained in DATA_LICENSE.txt. The complete source archive was fetched from the pinned author commit; archive SHA-256 and per-image file/pixel hashes are recorded. Only small illustrative figures are republished, not the dataset archive. Stable Diffusion images are supplied by the dataset authors; this study did not run Stable Diffusion.

The GAN and detector code was independently written for this controlled experiment. A GAN discriminator learns to distinguish its training generator from its real distribution; this is not automatically a universal AI-image detector.
