This is a starting kit for the Remote Sensing Image Scene Classification. 
We use the NWPU-RESISC45 satelite dataset presented in an image processing paper. The data set contains 31500 of satelite images of 45 class, each having 700 images.

References and credits: 

Gong Cheng,  Junwei Han,  and Xiaoqiang Lu,  RemoteSensing  Image  Scene  Classification:   Benchmark  andState of the Art. IEEE International.

Prerequisites:
Install Anaconda Python 3.6.6 

Usage:

(1) If you are a challenge participant:

- The file README.ipynb contains step-by-step instructions on how to create a sample submission for the Inria Aerial Image Labeling challenge. 
At the prompt type:
jupyter-notebook README.ipynb

- modify sample_code_submission to provide a better model or you can also write your own model in the jupyter notebook.

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- download the public_data and run (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

then zip the contents of sample_result_submission (without the directory).

(2) If you are a challenge organizer and use this starting kit as a template, ensure that:

- you modify README.ipynb to provide a good introduction to the problem and good data visualization

- sample_data is a small data subset carved out the challenge TRAINING data, for practice purposes only (do not compromise real validation or test data)
