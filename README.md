Distracted Driver Detection with Computer Vision


**Project Overview**
This project is all about using computer vision and deep learning to figure out if a driver is paying attention to the road or getting distracted. It looks at pictures from a dashboard camera and puts the driver into one of four groups: paying attention, drinking coffee, checking the mirror, or messing with the radio. The code walks through the whole process, starting with basic neural networks and moving all the way up to advanced image models and transfer learning. It also includes tools to test how well the models work and to peek under the hood to see how they make their choices.

**How the Code Works**
The code is broken down into a few main sections that handle everything from loading the pictures to testing the final results.

**Prepping the Data**
The first part of the code handles getting the data ready to use. It reads the image labels from a text file and makes sure we have the exact same number of pictures for each of the four categories. This stops the model from favoring one type of picture just because it saw it more often. To help the model learn better, the code also changes the training pictures slightly. It uses a tool called imgaug to spin, stretch, zoom, and flip the images, and it even takes out some color channels. This forces the model to learn the actual shapes of the driver and the car instead of just memorizing the exact photos we gave it.

**Building the Models**
Next, the project tests out three different ways to build the neural network to see what works best.

The first try is a basic dense network. This one just flattens the picture out into a long line of pixels, which is a good starting point to set a baseline but is usually not great for understanding photos.

The second try is a custom convolutional neural network built from scratch.

This type of model is much better for pictures because it uses filters to sweep across the image and find edges, shapes, and patterns, like a hand holding a cup.

Finally, the code uses transfer learning. This means it takes heavy-duty models like VGG16 or ResNet50 that have already looked at millions of pictures and know how to see things. The code keeps the base of these smart models but changes their final layers so they focus only on finding our specific distracted driving categories.

**Testing and Checking**
Just looking at an overall accuracy number is not enough, especially for something related to driving safety. The code includes tools to graph how well the model learns over time so you can catch if it starts cheating by memorizing the training data.

It also changes the problem from four categories down to just two: distracted or paying attention. This lets the code build a confusion matrix.  A confusion matrix is a grid that helps us see exactly what kind of mistakes the model makes. For example, it tells us if the model wrongly thinks a distracted driver is actually paying attention, which is a very dangerous mistake to make.
