# SpectroGAN


>## Express emotion through Images


Contains code for SpectroGAN, USC EE599 Deep Learning for Engineers Spring 2020.

Github link: [https://github.com/hegde95/GAN-for-speech-spectrogram](https://github.com/hegde95/GAN-for-speech-spectrogram) <br/>



## Objective <br />
The main objective is to apply style transfer on speech spectrograms in order to change the emotions conveyed in said speech.<br/>

Recent studies have successfully shown how style transfer can be applied on images from one domain to another. In this project we attempt to use this technique to embed emotions in spectrogram images. The end goal of the project will be to show that speech audio recorded with the connotation of one emotion can be conveted to another emotion without changing the content/information convayed in the speech. <br />


## Methodology <br/>
### Data set: <br/>
For this project we chose the RAVDESS [RAVDESS](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/) data set. The data set contains emotions from anger, calm, disgust, fearful, happy, neutral, sad and surprised. The cleaned and re-arranged data can be found [here](https://drive.google.com/drive/folders/12o5dMpEHqxIb8Qm9yHZB0s9at2lw3KPM?usp=sharing)
<br />

### Patch GANs: <br/>
Initially we constructed an CycleGAN to implement style trasfer. For our project we attemted to implement a Patch GAN. The code was based on [this link](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/)<br />


