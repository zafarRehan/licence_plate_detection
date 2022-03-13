# Licence Plate Detection

This repository uses custom-trained MobileNet-SSD V2 model for Object Detection to detect Licence plates in Images

This model will only detect one class of object i.e. Licence Plates </br>
The explained training of the model can be found in my training repo here : https://github.com/zafarRehan/tensorflow_transfer_learning

<h2>Contents</h2>

<h3>1. exported-model:</h3> 
This folder contains the trained model that we got as the result from <a src="https://github.com/zafarRehan/tensorflow_transfer_learning">tensorflow_transfer_learning</a>

<h3>2. models:</h3> 
This folder is the official Tensorflow's models library https://github.com/tensorflow/models which is needed for custom training a tensorflow model and in this case loading the custom trained model. This repo also has cool drawing functionalities which I used for drawing the bounding boxes over image.

<h3>3. custom.pbtxt:</h3>         
This file contains the class labels and is the same file used while training the model. 

<h3>4. detect.py:</h3>       
The python file which consists the code to</br>
1. Load Model</br>
2. Detect Licences</br>
3. Draw the output over image 

<h2>Running the code</h2>
1. In order to run the code you must have tensorflow 2.2 or above intalled </br>
2. Make sure to change your present working directory to <b>path to repo folder + licence_plate_detection/models/research</b></br>
3. Run the following code:
    ```
    protoc object_detection/protos/*.proto --python_out=.
    ```
4. Change the BASE_PATH in line 10 of the code as guided in comment </br>
5. Proceed to run the code

</br></br>

The code is well-commented so more details can be found in comments

</br></br>

<h2>Input Images</h2>
<p>
<img src="/test1.jpg" width=350>  <img src="/test2.jpg" width=350>
</p>

</br></br>

<h2>Output Images</h2>
<p>
<img src="/test1_output.jpg" width=350>  <img src="/test2_output.jpg" width=350>
</p>

