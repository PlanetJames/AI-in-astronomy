# AI in Astronomy

The field of AI and deep learning is ever expanding and evolving, so can be daunting at first to figure out where to start and difficult to understand what might be of use to you. This repository provides links and information about Artificial Intelligence (AI), covering general concepts, how artificial neural networks (ANN) work, types of deep learning (DL), tools and packages, and examples of how AI has been used in the field of astronomy. It is non-exhaustive but can hopefully act as an initial guide.

This page is my own work, not created by generative AI: while nowadays generative AI can be asked to supply information on given topics and even provide references, it is still very generic in its responses and is often unreliable, sometimes even hallucinating information. In contrast, the information and links provided below come from my firsthand experience in getting to grips with AI as an astronomer, so are more specifically tailored to (and hopefully a better guide for) astronomy students going through a similar process.

For many explanations & examples of the things below, you can often just Google them and look for tutorials/articles by GeeksforGeeks (https://www.geeksforgeeks.org/deep-learning-tutorial/), TowardsDataScience (https://towardsdatascience.com/), or MachineLearningMastery (https://machinelearningmastery.com/), among others - see the [Guides](#guides) section. That said, I've done my best to provide here a coherent structure, including all the key terms that I've come across and giving overviews and explanations where necessary. The links below are all example guides that I've found useful but are not exhaustive - feel free to search for others!


## Contents

1. [AI in General](#ai-in-general)
    1. [General abbreviations](#general-abbreviations)
    2. [Example categories of AI usage](#example-categories-of-ai-usage)
    3. [Example paradigms of machine learning](#example-paradigms-of-machine-learning)
    4. [Overview of machine learning methods (not deep learning)](#overview-of-machine-learning-methods)
2. [Introduction to Deep Learning](#introduction-to-deep-learning)
    1. [Guides](#guides)
    2. [How a network learns](#how-a-network-learns)
    3. [Core concepts](#core-concepts)
    4. [Other aspects of deep learning](#other-aspects-of-deep-learning)
    5. [Extras](#extras)
3. [Tools and Packages](#tools-and-packages)
    1. [Python tutorials](#python-tutorials)
    2. [Python packages (each have guides on their own websites as well)](#python-packages)
    3. [Designing and visualising neural network architectures](#designing-and-visualising-neural-network-architectures)
    4. [Citizen science](#citizen-science)
4. [Types of Deep Learning](#types-of-deep-learning)
    1. [CNN-based data processing and analysis](#cnn-based-data-processing-and-analysis)
    2. [Generative AI](#generative-ai)
    3. [Miscellaneous](#miscellaneous)
5. [AI use in Astronomy](#ai-use-in-astronomy)
    1. [Summaries of ML in astronomy](#summaries-of-ml-in-astronomy)
    2. [Example usage](#example-usage)


## AI in General

Here, we'll start with a broad overview of what artificial intelligence, machine learning, and deep learning mean, as well as their broad usage categories (e.g. classification) and paradigms (e.g. supervised learning). If you instead want to go straight to guides and explanations of deep learning, see the [Introduction to Deep Learning](#introduction-to-deep-learning) section, or [Types of Deep Learning](#types-of-deep-learning) and [AI use in Astronomy](#ai-use-in-astronomy) for the various types of deep learning models and how they're used in astronomy. Tutorials and packages for deep learning are provided in the [Tools and Packages](#tools-and-packages) section.

### General abbreviations
https://en.wikipedia.org/wiki/Glossary_of_artificial_intelligence

  AI = Artifical Intelligence (_What is AI?_ https://www.ibm.com/topics/artificial-intelligence) \
  ML = Machine Learning (_What is ML?_ https://www.ibm.com/topics/machine-learning) \
  DL = Deep Learning (_AI vs. ML vs. DL vs. NN_ https://www.ibm.com/blog/ai-vs-machine-learning-vs-deep-learning-vs-neural-networks/) \
  (A)NN = (Artificial) Neural Network

**IBM AI articles** \
\- https://www.ibm.com/think/artificial-intelligence#:~:text=Listen%20now-,Articles \
\- _What is a chatbot?_ https://www.ibm.com/topics/chatbots

### Example categories of AI usage
- Classification
- Generative model
- Regression
- Clustering (e.g. hierarchical, k-means, DBSCAN)
- Dimension reduction
- Density estimation
- Anomaly detection
- Data cleaning
- Feature learning

### Example paradigms of machine learning
Supervised vs unsupervised learning - https://www.geeksforgeeks.org/supervised-unsupervised-learning/
- Supervised
  - Typically for classification & regression.
  - E.g. artificial neural networks (ANN); decision trees; linear regression; random forest (RF); support vector machines (SVM)
- Unsupervised
  - Used for tasks like clustering, anomaly detection and dimensionality reduction that do not require a loss function.
  - One subset is self-supervised learning (SSL) - used for classification and regression tasks typical to supervised learning, e.g. VAEs, GANs - https://www.ibm.com/topics/self-supervised-learning
- Semi-supervised
  - Make use of labelled and unlabelled data.
  - https://www.ibm.com/topics/semi-supervised-learning
  - https://www.v7labs.com/blog/semi-supervised-learning-guide
- Reinforcement learning (RL)
  - Have an "agent" that can learn from trial and error in an interactive environment.
  - https://www.geeksforgeeks.org/what-is-reinforcement-learning/
  - https://towardsdatascience.com/reinforcement-learning-101-e24b50e1d292

### Overview of machine learning methods (not deep learning) <a name="overview-of-machine-learning-methods"></a>
- Machine Learning Tutorial: A Step-by-Step Guide for Beginners - https://www.simplilearn.com/tutorials/machine-learning-tutorial
- Understand Machine Learning Algorithms - https://machinelearningmastery.com/start-here/#:~:text=Your%20Python%20Projects-,Understand,-Machine%20Learning%20Algorithms


## Introduction to Deep Learning

Below are some general guides to help get you started with deep learning, including their history, architecture, learning process, various types, and methods to improve them. After that, the rest of this section goes into detail about the parts that make up a neural network and how such networks learn. If you want to get straight to trying out training a network yourself, take a look at the [Tools and Packages](#tools-and-packages) section below. If you want to learn more about the various types of deep learning models (like CNNs, language models, and generative AI), see the [Types of Deep Learning](#types-of-deep-learning) section. Or if you want to see how these models are used in astronomy, skip to the [AI use in Astronomy](#ai-use-in-astronomy) section at the end.

### Guides

#### History of deep learning (DL), overview of DL architectures, and how they learn
- Deep Learning Neural Networks Explained in Plain English - https://www.freecodecamp.org/news/deep-learning-neural-networks-explained-in-plain-english/

#### Overview of DL architectures, how they learn, and network types
- v7labs: The Essential Guide to Neural Network Architectures - https://www.v7labs.com/blog/neural-network-architectures-guide
- Machine Learning Glossary - https://ml-cheatsheet.readthedocs.io/en/latest/index.html
- GeeksforGeeks Deep Learning Tutorial (including example code) - https://www.geeksforgeeks.org/deep-learning-tutorial/

#### "Deep dive" into DL: overview, architectures, tutorials, improvements
- Deep Learning (scroll down for more categories) - https://machinelearningmastery.com/start-here/#:~:text=Customised%20Ensemble%20Algorithms-,Deep%20Learning

#### Resources for models and tutorials (also see [Tools and Packages](#tools-and-packages))
- HuggingFace, a platform where the machine learning community collaborates on models, datasets, and applications - https://huggingface.co/
- Papers With Code, a free and open resource with machine learning papers, code, datasets, methods and evaluation tables - https://paperswithcode.com/

### How a network learns

Neural networks consist of interconnected layers of artificial neurons, each made up of **weights and biases** (stored as matrix elements; typically randomised at first unless using pre-trained values (known as transfer learning)). These weights and biases are combined with each neuron's input (such as one or more pixels of an image) via a pre-defined **non-linear "activation" function** to produce that neuron's output. In this way, an input to the network (such as an image) will be fed through the network to produce an output (such as a binary "yes/no" classification), and we want the network to give the correct output each time. For supervised learning, we seek to train the network to minimise the difference between these outputs and the ground truths across all inputs (aka fitting the data). During training, a **"gradient descent" optimiser algorithm** combined with a defined learning rate allow the network to optimise a **cost (or "loss") function**, which is a predefined function that compares the output of the network to the ground truth for each input (e.g. mean square error). A process known as **backpropagation** then allows this error in the output to be propagated backwards through the network so the weights and biases can be suitably adjusted. Through repeating this process for each input or batch of inputs, the network "learns" what to output. There are various types of network architectures, layers, activation functions, optimisers, and loss functions, depending on the task you want it to do.

### Core concepts

#### The learning process
- Overview of a Neural Network’s Learning Process - https://medium.com/data-science-365/overview-of-a-neural-networks-learning-process-61690a502fa
- Weights & Biases
  - https://towardsdatascience.com/whats-the-role-of-weights-and-bias-in-a-neural-network-4cf7e9888a0f
  - https://www.geeksforgeeks.org/the-role-of-weights-and-bias-in-neural-networks/
- Activation function - https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
- Gradient descent & learning rate - https://www.ibm.com/topics/gradient-descent
- Specific optimizers - https://medium.com/mlearning-ai/optimizers-in-deep-learning-7bf81fed78a0
- Cost/loss functions - https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/
- Common loss functions - https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
- Backpropagation - https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd

**Hyperparameters** like learning rate, layer width & depth, and input batch size often require **fine-tuning** as well. Hence we split our data set into three: **training, validation, and test sets**.
- Data set types - https://towardsdatascience.com/training-vs-testing-vs-validation-sets-a44bed52a0e1
- Data set types - https://www.v7labs.com/blog/train-validation-test-set
- Training: batch size & epochs - https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
- Cross-validation
  - https://machinelearningmastery.com/k-fold-cross-validation/
  - https://towardsdatascience.com/cross-validation-a-beginners-guide-5b8ca04962cd
- Fine-tuning - https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8
- Fine-tuning can be automated, training & validating the model repeatedly, typically using either a grid search or random search approach to exploring the hyperparameters, or using an more advanced Bayesian optimisation method.
  - https://medium.com/@linmarsirait2/hyperparameter-tuning-in-machine-learning-using-bayesian-optimization-8ee522ef6d99
  - https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f

#### Evaluating the performance of classifiers
- Precision, recall, sensitivity and specificity - https://towardsdatascience.com/should-i-look-at-precision-recall-or-specificity-sensitivity-3946158aace1
- Confusion matrix - https://www.v7labs.com/blog/confusion-matrix-guide
- Area Under the Curve (AUC) ROC curve - https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

#### Evaluating training & validation performance
- Learning curve - https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
- Overfitting vs underfitting - https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/

### Other aspects of deep learning

#### Transfer learning
- https://machinelearningmastery.com/transfer-learning-for-deep-learning/
- https://cs231n.github.io/transfer-learning/

#### Foundation models
- https://www.datacamp.com/blog/what-are-foundation-models
- https://aws.amazon.com/what-is/foundation-models/
- https://medium.com/@tenyks_blogger/the-foundation-models-reshaping-computer-vision-b299a91527fb
- _On the Opportunities and Risks of Foundation Models_ - https://arxiv.org/abs/2108.07258

#### Learning with humans
These methods can also utilise crowdsourcing / citizen science as the "human" part, e.g. using human-made classifications to train a neural network.
- Human-in-the-loop
  - https://medium.com/vsinghbisen/what-is-human-in-the-loop-machine-learning-why-how-used-in-ai-60c7b44eb2c0
- Active learning
  - https://www.geeksforgeeks.org/ml-active-learning/
  - https://towardsdatascience.com/active-learning-in-machine-learning-525e61be16e5

### Extras
You may want to come back to these once you've had a go training a network for yourself first, in which case take a look at the [Tools and Packages](#tools-and-packages) section below.

#### Modifications to (hopefully) improve performance
- Addressing overfitting with Regularization - https://www.geeksforgeeks.org/regularization-in-machine-learning/
- Addressing overfitting with Dropout - https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9
- Skip Connections (used in ResNets, DenseNets, U-Nets) - https://theaisummer.com/skip-connections/
- How To Improve Deep Learning Performance - https://machinelearningmastery.com/improve-deep-learning-performance/

#### (Stochastic) Bayesian Neural Networks for predicting uncertainties
- A Tutorial for Deep Learning Users - https://arxiv.org/abs/2007.06823
- https://towardsdatascience.com/why-you-should-use-bayesian-neural-network-aaf76732c150
- https://keras.io/examples/keras_recipes/bayesian_neural_networks/
- https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd
- Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference - https://arxiv.org/abs/1506.02158
- Uncertainties in Parameters Estimated with Neural Networks: Application to Strong Gravitational Lensing - https://iopscience.iop.org/article/10.3847/2041-8213/aa9704/meta

#### Understanding and interpretability
- Occlusion Mapping
  - https://towardsdatascience.com/inshort-occlusion-analysis-for-explaining-dnns-d0ad3af9aeb6
- Attention Mapping
  - https://towardsdatascience.com/learn-to-pay-attention-trainable-visual-attention-in-cnns-87e2869f89f1
- Grad-CAM (for CNNs) - Gradient-weighted Class Activation Mapping
  - Original Grad-CAM paper - https://arxiv.org/abs/1610.02391
  - https://github.com/jacobgil/pytorch-grad-cam
  - https://towardsdatascience.com/understand-your-algorithm-with-grad-cam-d3b62fce353
- Google DeepDream
  - DeepDream Explained Clearly - https://www.garysnotebook.com/20190826_1


## Tools and Packages

Here you will find various tutorials, tools and packages for deep learning to help get you started with training your own model. You may want to consult the following [Types of Deep Learning](#types-of-deep-learning) and [AI use in Astronomy](#ai-use-in-astronomy) sections to learn about the various types and their applications to help you choose an appropriate model. Basic CNNs are often the go-to for a first model to try out, especially with the prevelance of image data sets like MNIST (which can be loaded via deep learning python packages or downloaded manually).

### Python tutorials
- Your First Machine Learning Project in Python Step-By-Step - https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
- Simple AI articles (2017 intro to ML) - https://medium.com/simple-ai
- (Also see "deep dive" guide above.)

### Resources for models and tutorials
- **HuggingFace**, a platform where the machine learning community collaborates on models, datasets, and applications - https://huggingface.co/
- **Papers With Code**, a free and open resource with machine learning papers, code, datasets, methods and evaluation tables - https://paperswithcode.com/

### Python packages
(Each have guides on their own websites as well.)
- Machine learning and data pre-processing (e.g. PCA, clustering, component separation, Random Forests):
  - **Scikit-learn** - https://scikit-learn.org/stable/
- Deep learning: \
  (There are various articles comparing these packages, e.g. https://www.simplilearn.com/keras-vs-tensorflow-vs-pytorch-article)
  - **Tensorflow** - https://www.tensorflow.org/
    - Introduction to Tensorflow: https://machinelearningmastery.com/introduction-python-deep-learning-library-tensorflow/
  - **Keras** (uses Tensorflow) - https://keras.io/
    - Introduction to Keras: https://machinelearningmastery.com/start-here/#:~:text=Customised%20Ensemble%20Algorithms-,Deep%20Learning%20(Keras)
    - Tensorflow & Keras: https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
  - **PyTorch** - https://pytorch.org/
    - Introduction to PyTorch: https://machinelearningmastery.com/start-here/#:~:text=Python%20with%20Keras-,Deep%20Learning%20(PyTorch)
- Deep probabilistic programming/modelling:
    - **PyRo** - https://pyro.ai/
- Efficient library (for speed and automatic support for GPUs):
  - **JAX** - https://theaisummer.com/jax/
- Active learning:
  - **AstronomicAL** (a human-in-the-loop interactive labelling and training dashboard for visualisation, integration and classification of data using Active Learning) - https://github.com/grant-m-s/AstronomicAL
- Bayesian Hyperparameter Optimisation (there's probably many more recent methods as well):
  - GPyOpt - https://sheffieldml.github.io/GPyOpt/
  - HyperOpt - https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0

### Designing and visualising neural network architectures
- https://datascience.stackexchange.com/questions/14899/how-to-draw-deep-learning-network-architecture-diagrams
- https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network

### Citizen science
- Zooniverse (crowdsourced data mining) - https://www.zooniverse.org/
- Zooniverse ML integration (including active learning) - https://git.astron.nl/astron-sdc/escape-wp5/workflows/zooniverse-integrating-machine-learning


## Types of Deep Learning

### CNN-based data processing and analysis

The basic CNN architecture has proved enduring for it's ability to extract features from images for a variety of tasks (generally image classification). As such, the CNN model has been a backbone for creating both larger models with greater complexity and performance (e.g. ResNet, EfficientNet), as well as being adapted in other ways to handle other image-based tasks (e.g. U-Net image segmentation).

**CNN - Convolutional Neural Network** (typically supervised; used when working with images; building block for many other models) \
  CNNs are most often used for classification problems, though they can also be used for regression tasks. \
  https://towardsdatascience.com/an-introduction-to-convolutional-neural-networks-eb0b60b58fd7
- CNN Layer Types:
  - Convolutional layers (feature extraction from input images; contains convolutional filters/kernels)
  - Pooling layers (performs down sampling: decrease the size of input images so later layers see more abstract features) \
    https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/
  - Fully connected ("dense") layers (draw relationships between extracted features to classify the input)
- Notes on appropriate convolutional kernel size: \
  https://towardsdatascience.com/deciding-optimal-filter-size-for-cnns-d6f7b56f9363 \
  https://medium.com/analytics-vidhya/how-to-choose-the-size-of-the-convolution-filter-or-kernel-size-for-cnn-86a55a1e2d15
- CNNs for regression (example given in Matlab rather than Python, but may still be useful for understanding): \
  https://uk.mathworks.com/help/deeplearning/ug/train-a-convolutional-neural-network-for-regression.html
- Example advanced CNN architectures: \
  https://www.geeksforgeeks.org/convolutional-neural-network-cnn-architectures/
  - **GoogleNet / Inception** (inputs pass through modules, each containing multiple different sized filters) \
    https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202
  - **ResNet - Residual Network** (uses "skip connections" to allow more robust training of deeper networks) \
      _Deep Residual Learning for Image Recognition_ - https://arxiv.org/abs/1512.03385 \
      https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff \
      https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
    - Understanding residual blocks \
      https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec \
      https://paperswithcode.com/method/residual-block
  - **DenseNet** (all layers interconnected, like a ResNet, but slightly different connection for better results) \
    Compared to ResNet, it has higher accuracy & requires fewer parameters, but is more memory-intensive and has more hyperparameters to tune. \
    https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803
  - **EfficientNet(V2)** (very high-performing CNN using "compound scaling") \
    Compound scaling is the systematic scaling of the model's dimensions (width, depth, and resolution) according to the input shape to produce the optimal network architecture. EfficientNetV2 improves performance using MBConv blocks (inverted residual blocks) among other things.
    - EfficientNetV1 \
      https://paperswithcode.com/method/efficientnet \
      https://arjun-sarkar786.medium.com/understanding-efficientnet-the-most-powerful-cnn-architecture-eaeb40386fad
    - EfficientNetV2 \
      https://paperswithcode.com/method/efficientnetv2 \
      https://medium.com/aiguys/review-efficientnetv2-smaller-models-and-faster-training-47d4215dcdfb
	
**3D CNN** (typically supervised; uses three-dimensional filters & pooling to deal with 3D inputs, but is notably slower than conventional 2D CNNs) \
  https://medium.com/@saba99/3d-cnn-4ccfab119cc2 \
  https://medium.com/@tvial_77168/convolutional-neural-networks-in-a-3d-world-30d66b304bfd \
  https://towardsdatascience.com/step-by-step-implementation-3d-convolutional-neural-network-in-keras-12efbdd7b130

**R-CNN - Regional CNN** (typically supervised; used for object detection - in addition to classification, it also applies a bounding box around each identified object, so can identify multiple objects of various classes) \
  https://www.geeksforgeeks.org/r-cnn-region-based-cnns/ \
  https://medium.com/analytics-vidhya/region-based-convolutional-neural-network-rcnn-b68ada0db871

**U-Net** (typically supervised; mainly for image segmentation, though has also been used for denoising, super-resolution, and more) \
  U-Nets contain an encoder and decoder, but unlike autoencoders the decoder also makes use of encoder information via skip connections. These skip connections aid in accurate feature extraction, but prevent the encoder and decoder from being separated. \
  https://www.geeksforgeeks.org/u-net-architecture-explained/ \
  https://towardsdatascience.com/understanding-u-net-61276b10f360

### Generative AI

All generative AI models are trained to inherently generate new data (e.g. text, images) from randomness, using different techniques to do so. Variational Autoencoders (VAEs) were the first type of generative AI, but often produced somewhat blurry images. Generative Adversarial Networks (GANs) have also been used for generating sharper images, particularly for specific types of images (e.g. faces), though they are difficult to train and hence not well suited for more general image generation. The first of the more general generative AI (e.g. DALL-E 1) used a combination of Transformers and VAEs to split text prompts into tokens, encode these as embeddings (representations of the tokens that machines can understand), and then decode these into new images. The latest text-to-image systems (e.g. DALL-E 2, DALL-E 3, Stable Diffusion, Imagen, MidJourney) now use a different approach - diffusion models.

For natural language processing (NLP) tasks, recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks were used but processed words one by one. Modern Large Language Models (LLMs) like ChatGPT use a transformer (GPT itself stands for Generative Pretrained Transformer), which can process all the words in a sentence simultaneously to speed up training, as well as to learn the relationships between words to provide context and meaning.

https://tryolabs.com/blog/2022/08/31/from-dalle-to-stable-diffusion \
https://research.ibm.com/blog/what-is-generative-AI

**A Few Articles on the Ethics of Generative AI** \
\- https://towardsdatascience.com/generative-ai-ethics-b2db92ecb909 \
\- https://blogs.ed.ac.uk/ede/2023/07/13/generative-ai-ethics-all-the-way-down/ \
\- https://www.forbes.com/sites/forbestechcouncil/2023/10/17/which-ethical-implications-of-generative-ai-should-companies-focus-on/ \
\- https://link.springer.com/article/10.1007/s10676-024-09745-x (Klenk 2024. _Ethics of generative AI and manipulation: a design-oriented research agenda_)

**AE - Autoencoder** (unsupervised/self-supervised; consists of an encoder and decoder, used to learn a latent space representation by regenerating the input image) \
  Autoencoders have a wide range of applications, including data compression, dimensionality reduction, anomaly detection, facial recognition, denoising, and generative tasks. The encoder and decoder can be separated after training, e.g. allowing the decoder to be used for image generation. \
  https://www.ibm.com/topics/autoencoder \
  https://medium.com/@soumallya160/the-basic-concept-of-autoencoder-the-self-supervised-deep-learning-454e75d93a04 \
  https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726
- Variants:
  - **VAE - Variational Autoencoder** (a version that learns probability distributions for their latent space, making them a form of stochastic encoding (rather than deterministic) and hence a type of generative AI - for example, VAEs were used for OpenAI’s original Dall-E image generation model)

**GAN - Generative Adversarial Network** (typically unsupervised; consists of a "generator" and a "discriminator" with the aim of genenerating new images, mainly for specific use cases such as face generation) \
  https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/ \
  https://www.geeksforgeeks.org/generative-adversarial-network-gan/ \
  https://medium.com/@marcodelpra/generative-adversarial-networks-dba10e1b4424 \
- GAN vs AE: When it comes to generative AI, GANs are typically more difficult to train than autoencoders, but produce sharper images. \
  https://medium.com/@parakatta/vae-v-s-gan-a-case-study-b09c7169ac02
- Variants:
  - **cGAN - conditional GAN** (trained with labels so it can later generate images with specific details) \
    https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

**RNN - Recurrent Neural Network** (typically supervised; processes sequential or time series data, such as for Natural Language Processing) \
  Applied to e.g. NLP, speech recognition, language translation. Has a form of "memory" so that, for a given input, prior inputs can be incorporated to influence the output (e.g. earlier words in a sentence combine to draw meaning/context for the current word being processed). RNN training uses a backpropagation through time (BPTT) algorithm, which is akin to backpropagation but with the additional summing of errors at each time step. \
https://www.ibm.com/topics/recurrent-neural-networks \
https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/ \
https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Variants:
  - **LSTM - Long Short-Term Memory network** (popular as they incorporate "cells" in their hidden layers that help prevent them from forgetting contextual information from far earlier in an input sequence) \
    https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/ \
    https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21 \
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
  - **GRU - Gated Recurrent Unit** (networks that are similar to LSTMs, using "hidden states" instead of cells to address the same problem) \
    https://medium.com/@prudhviraju.srivatsavaya/lstm-vs-gru-c1209b8ecb5a
  - **BRNN - Bidirectional Recurrent Neural Network** (also takes into account future inputs when processing a given input (e.g. later words in a sentence can help draw additional meaning/context), but are more complex so more computationally expensive to train and more prone to overfitting) \
    https://www.geeksforgeeks.org/bidirectional-recurrent-neural-network/

**Transformer** (typically self-supervised; applied to text (or images - see ViT below); used for e.g. Natural language processing (NLP), such as in ChatGPT and DALL-E 1 (a combination of a transformer and VAE)) \
  Consists of an encoder & decoder to respectively encode text prompts into token embeddings and decode them to generate new sequences. Unlike other methods, transformers use Attention, i.e. "soft" weights that can change during testing, as well as "contextual embeddings" so that the encoding of each input element depends on both the element itself and its context in the input as a whole. This allows them to, for example, learn meaning through the relationships between words in a sentence, making them perfect for text encoding and processing input sequences. \
https://www.geeksforgeeks.org/getting-started-with-transformers/ \
https://blogs.nvidia.com/blog/what-is-a-transformer-model/ \
https://medium.com/@faheemrustamy/vision-transformers-vs-convolutional-neural-networks-5fe8f9e18efc \
https://machinelearningmastery.com/the-transformer-attention-mechanism/ \
https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f \
https://blog.codewithdan.com/the-abcs-of-ai-transformers-tokens-and-embeddings-a-lego-story/
- RNN vs LSTM vs Transformer: Transformers process inputs in parallel, making them faster than RNNs and LSTMs, and their Attention mechanism provides interpretability, though they may have higher computational & memory requirements. \
  https://medium.com/@mroko001/rnn-vs-lstm-vs-transformers-unraveling-the-secrets-of-sequential-data-processing-c4541c4b09f
- Variants:
  - **BERT model - Bidirectional Encoder Representations from Transformers** (processes text sequences both forward and backward to improve performance) \
    https://samanemami.medium.com/bert-bidirectional-encoder-representations-e98833f9dfcd \
    https://towardsdatascience.com/bert-3d1bf880386a
  - **GPT model - Generative Pretrained Transformer** (large language model pretrained on a vast amount of data) \
      https://towardsdatascience.com/gpt-model-how-does-it-work-74bbcc2e97d1 \
      https://www.linkedin.com/pulse/understanding-transformer-architecture-chatgpt-built-rastogi-lion-
  - **ViT - Vision Transformer** (i.e. transformers applied to images) \
      https://theaisummer.com/vision-transformer/ \
      https://medium.com/machine-intelligence-and-deep-learning-lab/vit-vision-transformer-cc56c8071a20 \
      https://medium.com/@hansahettiarachchi/unveiling-vision-transformers-revolutionizing-computer-vision-beyond-convolution-c410110ef061
    - **MaxViT - Multi-Axis Vision Transformer** (family of hybrid (CNN + ViT) image classification models)
        https://arxiv.org/abs/2204.01697 \
        https://github.com/google-research/maxvit (official Google repository) \
        https://github.com/ChristophReich1996/MaxViT (unofficial PyTorch implementation)
    - **Swin (Shifted Window) Transformer** (uses a hierarchical architecture with 'shifting windows' to address long-range dependencies within the data and increase efficiency) \
        https://chautuankien.medium.com/explanation-swin-transformer-93e7a3140877 \
        _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows_ - https://arxiv.org/abs/2103.14030
        _Swin Transformer V2: Scaling Up Capacity and Resolution_ - https://arxiv.org/abs/2111.09883

**Diffusion model** (aka score-based generative model, e.g. DALL-E 2, DALL-E 3, Stable Diffusion, Imagen, MidJourney) \
  These models iteratively add carefully controlled noise to the input over N steps. The model is trained to predict this added noise at each step. Small steps ensure that the final image can be approximated by the same underlying noise distribution that is being sampled from. Sampling from this distribution and beginning with a noisy image, this process is then reversed, gradually denoising the image over N steps to generate new data that resembles the input. The architecture for this is based on a U-Net thanks to it having the same input and output shapes to judge how much noise has been added. An autoencoder can also be used to upscale the final image, allowing the diffusion model to work with computationally inexpensive low-resolution images. \
  After training in this way on a large data set, we can supply the denoiser with different noise distributions to generate different images. By default this is unconditional based image generation, creating random images based on the training set. If text labels are supplied during training, these can be encoded using a language model and used to guide the diffusion model in generating specific images - this is conditional based image generation. For example, DALL-E 2 uses a technique called Guided Language to Image Diffusion for Generation and Editing (GLIDE), while Stable Diffusion uses a transformer. Other inputs can also be used instead of labels - for example, models that are supplied with segmentation maps to remove objects from images. \
https://shyampatel1320.medium.com/introduction-to-diffusion-models-and-imagen-the-magic-behind-text-to-image-generation-24221532580d \
https://tryolabs.com/blog/2022/08/31/from-dalle-to-stable-diffusion#:~:text=How%20does%20diffusion%20work%3F \
https://www.paepper.com/blog/posts/how-and-why-stable-diffusion-works-for-text-to-image-generation/

### Miscellaneous

**PINN - Physics Informed Neural Network** (supervised; use a physically-derived loss function so that output parameters correspond to physical variables) \
	https://towardsdatascience.com/physics-informed-neural-networks-pinns-an-intuitive-guide-fff138069563

**GNN - Graph Neural Network** (supervised, semi-supervised, or unsupervised; created to handle graphs, i.e. relationships between vertices/nodes and the lines/edges that connect them) \
	https://www.v7labs.com/blog/graph-neural-networks-guide \
 	https://blogs.nvidia.com/blog/what-are-graph-neural-networks/ \
  	https://www.geeksforgeeks.org/what-are-graph-neural-networks/
 
**RIM - Restricted Boltzmann Machine** (unsupervised; 2-layer fully connected network) \
	https://www.geeksforgeeks.org/restricted-boltzmann-machine/

**SOM - Self-Organizing Map** (unsupervised; used for for clustering and dimensionality reduction; produces a low-dimensional (typically 2D) representation of a higher dimensional data set while preserving the topological structure of the data) \
	https://towardsdatascience.com/self-organizing-maps-1b7d2a84e065 \
	https://www.geeksforgeeks.org/self-organising-maps-kohonen-maps/


## AI use in Astronomy

CNNs and their variants (e.g. ResNets, Mask R-CNN) are primarily used for detection and classification within astronomical images.
Recurrent networks are generally used for detection and classification within transient data (e.g. supernovae).
GANs often used for generating synthetic data, typically catalogues of images for training & testing other methods.
VAEs & U-Nets used for image modification (denoising, deblurring, etc.) or segmentation (e.g. deblending).

### Summaries of ML in astronomy

- _The Role of Machine Learning in the Next Decade of Cosmology_ (see p4) https://arxiv.org/abs/1902.10159
- _Foreword to the Focus Issue on Machine Learning in Astronomy and Astrophysics_ https://arxiv.org/abs/1906.08349
	(See Tables 1 & 2, plus Section 3 for examples of Emerging, Progressing, and Established uses of ML in various sub-fields)
- _Astronomia ex machina: a history, primer, and outlook on neural networks in astronomy_ https://arxiv.org/abs/2211.03796

### Example usage

Below are some example areas in astronomy where AI has been applied. There are far too many research publications to list here, though the occassional paper is provided for some of the more specific use cases.

- _Machine Learning and Artificial Intelligence applied to astronomy 2_ conference (Open University, 2021)
	https://ras.ac.uk/events-and-meetings/ras-meetings/machine-learning-and-artificial-intelligence-applied-astronomy-2
	(See PDF attachments for attendees' individual uses of AI)
- [CNN] Spectroscopic/photometric redshift estimation
- [CNN] Finding high-z sources
- [CNN] (Lensed) Gravitational wave detection
- [CNN] Detecting exoplanets
- [CNN] Predicting galaxy spectra
- [CNN] Estimating cosmological parameters from simulated 3D dark matter distributions
- [CNN] Weak gravitational lensing shear map estimation and dark matter mapping
- [CNN] Galaxy properties: surface brightness fitting, morphology, bars, halo
  - E.g. Mike Walmsley's Zoobot model applied to Galaxy Zoo's DECaLS data and later DESI imaging, trained on citizen science classifications (https://academic.oup.com/mnras/article/509/3/3966/6378289 and https://academic.oup.com/mnras/article/526/3/4768/7283169)
- [R-CNN transfer learning] Galaxy feature detection: finding giant star-forming clumps (https://arxiv.org/abs/2312.03503)
- [CNN, GAN, VAE] Deblending galaxies
- [GAN] High-resolution synthetic galaxy generation
- [GAN] Simulating the cosmic web
- [cGAN] Anomaly detection for JWST Imaging (e.g. https://iopscience.iop.org/article/10.3847/2515-5172/acff65)
- [Autoencoder] Super-resolving telescope imaging (e.g. https://doi.org/10.1093/mnras/stab2195)
- [RNN] Supernovae & quasar classification
- [Custom] Image Restoration (denoising & deblurring) - network like an autoencoder but with residual layers instead of a latent space (https://arxiv.org/abs/2311.00186)
- [Physics-aware autoencoder] Self-supervised galaxy model fitting (https://arxiv.org/abs/1907.03957)
- [Transformer] Improving ADS searches with natural language processing (https://arxiv.org/abs/2112.00590)
- [Transformer] Galaxy detection and deblending (e.g. https://academic.oup.com/mnras/article/526/1/1122/7273850)
- [Transformer] AstroLLaMA large language model for astronomy, trained on arXiv abstracts (https://arxiv.org/abs/2309.06126)
- [CNN and variants] Detection and modelling of strong gravitational lenses, for example:
  - A Bayesian approach to strong lens finding using an ensemble classifier (citizen science, ResNet, CNN) (https://ui.adsabs.harvard.edu/abs/2023arXiv231107455H/abstract)
  - Finding and rank-ordering strong gravitational lenses with DenseNets (densely connected CNN) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.4188N/abstract) - DenseNets achieve comparable true positive rates but considerably lower false positive rates (when compared to residual networks; ResNets), so are recommended.
  - Detecting strong lenses: exploring interpretability and sensitivity to rare lensing configurations (https://doi.org/10.1093/mnras/stac562)
  - Strong lens modelling: comparing and combining Bayesian neural networks and parametric profile fitting (https://academic.oup.com/mnras/article/505/3/4362/6287584)
