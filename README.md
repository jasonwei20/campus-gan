# campus-gan

Generating images of college campuses with GAN. Uses tensorflow for GAN, pyTorch for analysis. 

I then pass generated and real images through a pretrained ResNet in pytorch and retrieve embeddings after the first, second, third, and fourth blocks. I plot the embeddings with tSNE and train k-means and SVM on the embeddings to evaluate how much clustering occurs.
