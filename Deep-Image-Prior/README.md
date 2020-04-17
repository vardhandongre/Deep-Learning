# Deep Image Prior 
(Reimplementation)

[Original Research](https://dmitryulyanov.github.io/deep_image_prior)



## Abstract 

The paper introduces a novel approach to image restoration with deep convolutional networks. CNNs have
shown great promise for image restoration tasks, but rely on large datasets to form image priors that will
assist in image restoration. Another approach has been to handcraft these image priors instead of learning
them from these large datasets. Deep Image Prior serves as a bridge between these two approaches, where
a CNN is able to learn a prior for natural images without learning from a dataset, and instead captures the
prior with the structure of the network. The structure of the network then serves as the handcrafted prior.
Instead of optimizing in the image space, the authors propose to optimize in the space of neural network
parameters, and then map the parameters to an image in the image space through the forward function of
the network fθ. In contrast to image priors, the input to fθ is fixed for the entire training process. The
network is trained to minimize loss with respect to an observed degraded image (target image).

### Results:
#### Super Resolution:
<img src="https://github.com/vardhandongre/Deep-Learning/blob/master/Deep-Image-Prior/images/final.png" height="400">
<table><tr><td><img src="https://github.com/vardhandongre/Deep-Learning/blob/master/Deep-Image-Prior/images/final.png" style="width:100%"></td><td><img src='knight.gif' style="width:60%"></td></tr><tr><td>Hamiltonian cycle in Dodecahedron</td><td><center>Kinght's tour of Chessboard </center></td></table>

#### The following is the link to detailed report of our work:
[Report](http://pdfviewer.softgateon.net/?state=%7B%22ids%22:%5B%221djCrAgouX3FmuynFQ0FvgNnzMq79BVRs%22%5D,%22action%22:%22open%22,%22userId%22:%22109664371285326200548%22%7D) 