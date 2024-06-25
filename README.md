# 🖌 Image Colorization Project 2.0

## 🌟 Highlights

- We fixed the problem of having desaturated images that was present in Image Colorization Project 1.0
- We managed to color actual historical images realistically

## ℹ️ Overview

Just like in Image Colorization Project 1.0, here we tackled the problem of colorizing black and white images in a realistic way. We tried to solve the problem following two different approaches: first, we framed the problem as a 
regression problem, then we framed it as a classification problem. <br>
The architecture we used for our model was a simplified version of a U-Net, which we believed was particularly fit for this task. We used a pretrained ResNet-50 as encoder of the U-Net, and added some transpose convolutional layers with Batch 
Normalization to perform the upsampling. The only difference between the two approaches was the output shape. <br>
In the first case, we took each RGB image, converted it into LAB color space, retained the Luminance channel and trained a CNN to predict the A and B channel values for each pixel. This approach gave good results and had reasonably low
training and inference times. Here the output shape was (224, 224, 2), so for each pixel we predicted two values, one for the A and one for the B channel. <br>
The classification approach was a bit more technical, and we took inspiration from the paper 'Colorful Image Colorization' by Zhang et al. (2016). We first quantized the AB value for each pixel, so that each pixel belonged to one of roughly 200 classes.
We then used a weighted categorical cross entropy to train the model and returned a 200 dimensional probability distribution on the classes for each pixel. We then tried different sampling techniques from this distribution (like picking the mode, 
the expected value ...), and finally chose the technique which yielded better results. Unfortunately, training such networks was very computationally expensive, so we had to reduce the training data.<br>
Overall, we found that the solving the problem as a regression was easier, more computationally efficient and also yielded better results, and we used one of the regression models to actually color historical images (see Appendix) with great success.

### ✍️ Authors

Alessandro Ardenghi, Rocco Giampetruzzi, Rolf Minardi

