### Venture Funding with Deep Learning

This notebook includes a risk assessment of the startups that have requested investment. The input and output layers of the model are connected by several neuron layers in a neural network. A venture fund will be able to identify which firms will be profitable thanks to the deep neural network we construct. An algorithm based on deep learning and neural networks is used to predict the success of certain applications.

## Technologies
The following packages from Python 3.7 are used in this project:

* Jupyterlab Notebook
* Pandas 
* Keras
* TensorFlow 

## Installation Guide
Before running the application first install the following dependencies.
```
pip install --upgrade tensorflow
python -c "import tensorflow as tf;print(tf.__version__)"
python -c "import tensorflow as tf;print(tf.keras.__version__)"
```
and then import the required Python Libraries.
```
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```
## Usage
Simply clone the repository and execute the forecasting_net_prophet.ipynb to run this program:

 venture_funding_with_deep_learning.ipynb
  
## Results
The results after running each models.

Original Model Results:
```
268/268 - 0s - loss: 0.5534 - accuracy: 0.7293 - 152ms/epoch - 567us/step
Loss: 0.5533835887908936, Accuracy: 0.7293294668197632
```
Alternative Model 1 Results:
```
268/268 - 0s - loss: 0.5880 - accuracy: 0.7223 - 245ms/epoch - 914us/step
Loss: 0.5880142450332642, Accuracy: 0.7223323583602905
```
Alternative Model 2 Results:
```
268/268 - 0s - loss: 0.7345 - accuracy: 0.4835 - 296ms/epoch - 1ms/step
Loss: 0.7345243096351624, Accuracy: 0.48349854350090027
```
## Contributors
Contributed by Nayana Narayanan.

## License
MIT License