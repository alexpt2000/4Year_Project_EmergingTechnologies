A simple web application that recognizes handwritten digits. 

Developed using [TensorFlow](https://www.tensorflow.org/) and the super simple [Keras](http://keras.io/) Library. 

Wrapped into a Webapp using [Flask](http://flask.pocoo.org/) Micro Framework.

![Screencast](Screencast/screencast.png)


# Requirements

## Install Anaconda

Anaconda is a freemium open source distribution of the Python and R programming languages for large-scale data processing, predictive analytics, and scientific computing, that aims to simplify package management and deployment.

[More Info...](https://www.anaconda.com/)

[Download and Install](https://www.anaconda.com/download/)

https://www.anaconda.com/download/


## Tensorflow - Installing with Anaconda

TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks.

[More Info...](https://www.tensorflow.org/)

```pip install --upgrade tensorflow```


## Keras
Keras is an open source neural network library written in Python. It is capable of running on top of MXNet, Deeplearning4j, Tensorflow, CNTK or Theano.

[More Info...](https://keras.io/)

```conda install -c conda-forge keras```


# To run it locally, first clone the directory. 

```git clone https://github.com/alexpt2000gmit/4Year_Project_EmergingTechnologies``` 

Next cd into the directory.

```cd 4Year_Project_EmergingTechnologies```

Then install the dependencies using pip.

```pip install -r requirements.txt```

To start the Flask Server,

```python app.py```

Enter a URL in the address bar of your browser,

```http://localhost:1111/```


