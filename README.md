# ImplementGAN
December 2016
### Summary
In this project, I reimplemented Generative Adversarial Networks on 3 datasets: 
1. A simple 1-d Gaussian distribution
2. MNIST dataset (vectorized)
3. MNIST dataset (images) - (implemented DCGAN)

### Details
Generative adversarial networks are fascinating models with their ability of mimicking a data distribution (and hence generating data) without need of complex structures like parametric models or Markov chains. They have their own drawbacks of keeping generative and discriminative network in balance during training, and hardship of quantitatively mea- suring their success assessing their performance. However, they generate more real-like images compared to other models.

In this project, first I applied GANs to a Gaussian distribution. It was hard to find right parameters and even then, results were not that plausible, especially when the real data had large standard deviation since generator network was producing ”narrow” distributions compared to the real data. This is a known problem of GANs and attempts to fix this problem, as well as some others, is explained in the work of Salimans et al. [3].

When I apply GANs on MNIST data, I obtained more plausible results, since almost all of the generated figures were obvious to identify as the intended digit. The results presented here are not obtained with networks trained for very long time due to computational power restrictions. Because of the lack of a numerical comparison of results, I was not able to fine-tune the parameters. As future work, increasing training epochs and fine-tuning would give better results.

Comparing inputs of vectorized images and DC-GAN model where the images are fed to a convolutional network, although the training time was ~1000 times longer, results were worse and it was hard to identify generated digits. However, this is not a fair comparison since the advanced DC-GAN network did not have a chance to be fully trained with such less epochs and I could not afford to train them longer.

### Code References
Some of the code in this repo is obtained from the below sources and modified accordingly:
1. https://github.com/AYLIEN/gan-intro
2. https://github.com/carpedm20/DCGAN-tensorflow

### References
[1] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio, “Generative adversarial nets,” in Advances in Neural Information Processing Systems, 2014, pp. 2672–2680.

[2] Alec Radford, Luke Metz, and Soumith Chintala, “Un- supervised representation learning with deep convolu- tional generative adversarial networks,” CoRR, vol. abs/1511.06434, 2015.

[3] Tim Salimans, Ian J. Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen, “Improved techniques for training gans,” CoRR, vol. abs/1606.03498, 2016.

[4] YannLeCun,CorinnaCortes,andChristopherJCBurges, “The MNIST database of handwritten digits,” 1998.
