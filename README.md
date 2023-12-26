## Suicide Detection From User Input (Persian and English currently)

<h4>In this project we are using different machine learning and deep learning model to detect if a message is suicidal or not</h4>
<hr>
<h3>Project Structure</h3>
<ul>
<li>api/ : Api folder contains the api using FastAPI</li>
<li>dataset/ : You can put your dataset here. <a href="https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?resource=download">link</a></li>
<li>lib/ : This folder contains some useful modules</li>
<li>inference.py : Script to predict and classify using pre-trained model</li>
<li>model.joblib : Pre-Trained model using naive bayes classification</li>
<li>requirements.txt : Requirements file</li>
<li>train_nb.py : Script to train model</li>
</ul>
<hr>

<h3>Train</h3>
To train the dataset based on your own specifications you can go through the following steps:
<ol>
<li>Download dataset and place it in the dataset folder</li>
<li>Create your virtual environment using: <em>python venv</em></li>
<li>Install requirements using the command: <em>pip install requirements.txt</em></li>
<li>Change parameters such as <em>max_features</em> in text_processing.py</li>
<li>Run <em>train_nb.py</em> and save your model</li>
</ol>

<hr>
<h3>Predict</h3>
To predict you can use pre-trained model and run <em>
inference.py</em>

<hr>
<h3>Future Works</h3>
<ul>
<li>Train model using deep learning methods</li>
<li>Deploy api</li>
<li>Enhance translation and add more languages</li>
</ul>

