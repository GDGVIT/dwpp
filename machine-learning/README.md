# Machine Learning Algorithm

## How does it work?
The recommendation system is based on MobileNet Architecture. MobileNet is a type of convolutional neural network designed for mobile and embedded vision applications. Features are extracted from the feature layer and stored in a vector. The vector is averaged out each time a user likes a picture and with time an average vector representing their likings is formed. This helps us compare two pictures for recommendation.

The Architecture diagram is given below:

![MobileNet Architecture](https://miro.medium.com/max/1384/1*7R068tzqqK-1edu4hbAVZQ.png
"MobileNet Architecture")

To run the API with the recomenedation algorithm, run the following command in your shell to install the required dependencies.

`pip install -r requirements.txt`