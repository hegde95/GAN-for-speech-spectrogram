# SpectroGAN


>## Express emotion through Images


Contains code for SpectroGAN, USC EE599 Deep Learning for Engineers Spring 2020.

Github link: [https://github.com/hegde95/GAN-for-speech-spectrogram](https://github.com/hegde95/GAN-for-speech-spectrogram) <br/>



## Objective <br />
The main objective is to apply style transfer on speech spectrograms in order to change the emotions conveyed in said speech.<br/>

Recent studies have successfully shown how style transfer can be applied on images from one domain to another. In this project we attempt to use this technique to embed emotions in spectrogram images. The end goal of the project will be to show that speech audio recorded with the connotation of one emotion can be conveted to another emotion without changing the content/information convayed in the speech. <br />


## Methodology <br/>
### -- Data set: <br/>
For this project we chose the RAVDESS [RAVDESS](https://zenodo.org/record/1188976#.Xq-sIvJKg5k) data set. The data set contains lexically-matched statements in a neutral North American accent spoken with emotions from anger, calm, disgust, fearful, happy, neutral, sad and surprised. The cleaned and re-arranged data can be found [here](https://drive.google.com/drive/folders/12o5dMpEHqxIb8Qm9yHZB0s9at2lw3KPM?usp=sharing). For this project, we chose to convert audio from "calm" to "anger" and "fearful". 
<br />
### -- Data Conversion: <br/>
The source and target data format in this project are .wav files, but our GAN's work on images. 
1. **Audio to Image**: To convert the audio to spectrograms we sampled the audio at 16000 Hz and performed stft of lenght 512 and used a hop lenght of 256. The source audio files were also trimmed as to obtain a spectrogram of size 257 X 257. This image padded was with 0's to get a 260 X 260 array, which is the input and output to our GAN.
2. **Image to Audio**: To convert the generated spectrograms to audio, we used the [griffin-lim algorithm](https://www.researchgate.net/publication/261315209_A_Fast_Griffin-Lim_Algorithm) on the clipped image. We made sure that that the fft lenght and the hop lenght used in the istft was the same as before.<br />

### -- Patch GANs: <br/>
Initially we constructed an CycleGAN to implement style trasfer. For our project we attemted to implement a Patch GAN. The code was based on [this link](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/)<br />

## Results <br/>
Below are a couple of input and output audio files from calm to angry with the corresponding spectrograms. <br/>
[calm](/results/calm_orig2.wav)
[anger](/results/calm_orig2_Fearful_generated.wav)

<script src="/audiojs/audio.js"></script>
 <script>
   audiojs.events.ready(function() {
     var as = audiojs.createAll();
   });
 </script>
 
<audio src="/results/calm_orig2.wav" preload="auto" />


Here is the link to our presentation<br/>

Here is a video showing a demo<br/>

Here is a link to our report<br/>

## Contributors <br/>
Karkala Shashank Hegde - [https://www.linkedin.com/in/karkala-shashank-hegde/](https://www.linkedin.com/in/karkala-shashank-hegde/)<br/>
Vineeth Ellore - [https://www.linkedin.com/in/vineethellore/](https://www.linkedin.com/in/vineethellore/) <br/>
Ashwin Telagimathada Ravi - [https://www.linkedin.com/in/ashwin-tr/](https://www.linkedin.com/in/ashwin-tr/)<br/>
