#  Code used for "Neural and Complex Networks" subject at UNED

This repository contains code samples for the different
tasks made associated to the subject ["Redes Neuronales 
y Complejas"](https://portal.uned.es/portal/page?_pageid=93,71542273&_dad=portal&_schema=PORTAL&idAsignatura=2115612-&idTitulacion=215801)
at Universidad Nacional de Educación a Distancia (UNED) 
for the academic year 2022-2023.

Different unrelated tasks had to be done, so different 
subdirectories have been employed:

- 03_AttractorNetwork: for implementing a functional Hopfield model with different add-ons and variations to study their influence
- 07_FeedForwardNetwork: for studying the influence of the number of hidden layers in a Deep FeedForward Neural Network. This makes use of both the [repository of Michael Nielsen](https://github.com/mnielsen/neural-networks-and-deep-learning) and its [adaption to Python3 by Dobrzanski](https://github.com/MichalDanielDobrzanski/DeepLearningPython35)

For running the file main_deepnetwork.py a different environment 
has been created due to compatibility issues for the theano depreciated
library. That environment can be retrieved with the following 
dependencies in the poetry.toml file:

`[tool.poetry.dependencies]`

`python = "^3.9"`

`numpy = "1.20.3"`

`theano = "1.0.5"`