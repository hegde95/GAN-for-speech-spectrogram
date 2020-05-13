# SpectroGAN
<p align="center">
    <img src="/GAN-for-speech-spectrogram/results/gif.gif" alt="gif"/>
</p>

>## Express emotion through Images


Contains code for SpectroGAN, Final project for USC EE599 Deep Learning for Engineers Spring 2020.

Github link: [https://github.com/hegde95/GAN-for-speech-spectrogram](https://github.com/hegde95/GAN-for-speech-spectrogram) <br/>




## Objective <br />
The main objective is to apply style transfer on speech spectrograms in order to change the emotions conveyed in said speech.<br/>

Recent studies have successfully shown how style transfer can be applied on images from one domain to another. In this project we attempt to use this technique to embed emotions in spectrogram images. The end goal of the project will be to show that speech audio recorded with the connotation of one emotion can be conveted to another emotion without changing the content/information convayed in the speech. <br />


## Methodology <br/>
### -- Data set: <br/>
For this project we chose the RAVDESS [RAVDESS](https://zenodo.org/record/1188976#.Xq-sIvJKg5k) data set. The data set contains lexically-matched statements in a neutral North American accent spoken with emotions from anger, calm, disgust, fearful, happy, neutral, sad and surprised. The cleaned and re-arranged data can be found [here](https://drive.google.com/drive/folders/12o5dMpEHqxIb8Qm9yHZB0s9at2lw3KPM?usp=sharing). For this project, we chose to convert audio from "calm" to "anger" and "fearful". The entire set of npz files can be found at this links:<br/>

calm2surprised- https://drive.google.com/uc?id=15HlogMsEX9juzL1j7HqweDQv9F5tJFuG

calm2sad- https://drive.google.com/uc?id=15HlO9YvZjMtbcEiXajfE9uqmrVS0Unep

calm2happy - https://drive.google.com/uc?id=153PIrQEk_agKiUOP5cujrVyGjnxDqKhd

calm2fearful - https://drive.google.com/uc?id=14scuVs2nlNH29DIWecrNrCcwNAVR0orG

calm2disgust - https://drive.google.com/uc?id=14s7kWrDQP61X9QXYDV-W3W4YIukJs_55

calm2anger - https://drive.google.com/uc?id=14q4aZseMCQO_xbbmX-JRsbRSlGX9bB3E

fearful2surprised - https://drive.google.com/uc?id=167zknyKgV5r8qO_fLbbFLT1A76WwTWiL <br />

<br />

### -- Data Conversion: <br/>
The source and target data format in this project are .wav files, but our GAN's work on images. 
1. **Audio to Image**: To convert the audio to spectrograms we sampled the audio at 16000 Hz and performed stft of lenght 512 and used a hop lenght of 256. The source audio files were also trimmed as to obtain a spectrogram of size 257 X 257. This image padded was with 0's to get a 260 X 260 array, which is the input and output to our GAN.
2. **Image to Audio**: To convert the generated spectrograms to audio, we used the [griffin-lim algorithm](https://www.researchgate.net/publication/261315209_A_Fast_Griffin-Lim_Algorithm) on the clipped image. We made sure that that the fft lenght and the hop lenght used in the istft was the same as before.<br />

### -- CycleGANs: <br/>
For our project we attemted to implement a CycleGAN as this has been shown to perform well on style transfer tasks. Also, to be size (and therefore fft length) independent, we use a PatchGAN model for our descriminator network .This code was based on [this link](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/)<br />

## Results <br/>
Below are a couple of input and output audio files from calm to angry and fearful with the corresponding spectrograms. (***Click on the image to hear the audio***.) <br/>


### Audio from same data set:<br />

The following are results for 3, 6 and 9 ReNet blocks in the the transformer trained for 100 epochs:


|Emotion|"Dogs are sitting by the door" (3)|"Dogs are sitting by the door" (6)|"Dogs are sitting by the door" (9)|
| :----: | :----: | :----: | :----: |
|Neutral (Original)|[![Input Neutral speech](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01.wav)|
|Angry|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01_anger3_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01_anger3_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01_anger6_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01_anger6_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01_anger9_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/GoodAudio/neutral_01_anger9_generated.wav)|



The following are results for 260 X 260 and 520 X 520 spectrograms, trained for 100 epochs:

|Emotion|"Dogs are sitting by the door" (260 X 260)|"Dogs are sitting by the door" (520 X 520)|
| :----: | :----: | :----: |
|Calm (Original)|[![Input Neutral speech](/GAN-for-speech-spectrogram/results/GoodAudio/260_real_calm_for_anger.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/260_real_calm_for_anger.wav)|[![Input Neutral speech](/GAN-for-speech-spectrogram/results/GoodAudio/520_real_calm_for_anger.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/520_real_calm_for_anger.wav)|
|Angry|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/260_generated_anger.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/GoodAudio/260_generated_anger.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/520_generated_anger.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/GoodAudio/520_generated_anger.wav)|





|Emotion|"Kids are talking by the door"|"Dogs are sitting by the door"|"Dogs are sitting by the door"|"Dogs are sitting by the door"|
| :----: | :----: | :----: | :----: | :----: |
|Calm (Original)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08.wav)|
|Surprised|[![Output Surprised speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_surprised_generated.jpg "Output Surprised speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_surprised_generated.wav)|[![Output Surprised speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_surprised_generated.jpg "Output Surprised speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_surprised_generated.wav)|[![Output Surprised speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_surprised_generated.jpg "Output Surprised speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_surprised_generated.wav)|[![Output Surprised speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_surprised_generated.jpg "Output Surprised speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_surprised_generated.wav)|
|Fearful|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_fearful_generated.wav)|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_fearful_generated.wav)|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_fearful_generated.wav)|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_fearful_generated.wav)|
|Anger|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_anger_generated.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_anger_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_anger_generated.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_anger_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_anger_generated.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_anger_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_anger_generated.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_anger_generated.wav)|
|Disgust|[![Output Disgust speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_disgust_generated.jpg "Output Disgust speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_disgust_generated.wav)|[![Output Disgust speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_disgust_generated.jpg "Output Disgust speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_disgust_generated.wav)|[![Output Disgust speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_disgust_generated.jpg "Output Disgust speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_disgust_generated.wav)|[![Output Disgust speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_disgust_generated.jpg "Output Disgust speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_disgust_generated.wav)|
|Happy|[![Output Happy speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_happy_generated.jpg "Output Happy speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_happy_generated.wav)|[![Output Happy speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_happy_generated.jpg "Output Happy speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_happy_generated.wav)|[![Output Happy speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_happy_generated.jpg "Output Happy speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_happy_generated.wav)|[![Output Happy speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_happy_generated.jpg "Output Happy speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_happy_generated.wav)|
|Sad|[![Output Sad speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_sad_generated.jpg "Output Sad speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_sad_generated.wav)|[![Output Sad speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_sad_generated.jpg "Output Sad speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_11_sad_generated.wav)|[![Output Sad speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_sad_generated.jpg "Output Sad speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_14_sad_generated.wav)|[![Output Sad speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_sad_generated.jpg "Output Sad speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_08_sad_generated.wav)|



The above samples show that the spectrograms of angry and fearful speech have more predominant and spaced out harmonics. These are characteristics of angry and fearful speech.<br />


### Unseen audio from same data set:<br />

|Emotion|"Kids are talking by the door"|"Dogs are sitting by the door"|
| :----: | :----: | :----: |
|Calm (Original)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2.wav)|
|Angry|[![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Anger_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Anger_generated.wav)|
|Fearful|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Fearful_generated.wav)|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Fearful_generated.wav)|



### Same script by unseen actor:<br />

|Emotion|"Dogs are sitting by the door"|
| :----: | :----: |
|Calm (Original)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2.wav)|
|Angry|[![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Anger_generated.wav)|
|Fearful|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Fearful_generated.wav)|

### Lexically similar script by unseen actor:<br />


|Emotion|"This project is fun"|"Three plus one equals four"|
| :----: | :----: | :----: |
|Calm (Original)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3.wav)|
|Angry|[![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Anger_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Anger_generated.wav)|
|Fearful|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Fearful_generated.wav)|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Fearful_generated.wav)|


### Different laguage by unseen actor:<br />


|Emotion|"Gaadi waala aya ghar se kachra nikal"|"Konegu project mugithu"|
| :----: | :----: | :----: |
|Calm (Original)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin.jpg =257x "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin.wav)|[![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan.jpg =257x "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan.wav)|
|Angry|[![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Anger_generated.jpg =257x "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Anger_generated.wav)|[![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Anger_generated.jpg =257x "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Anger_generated.wav)|
|Fearful|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Fearful_generated.jpg =257x "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Fearful_generated.wav)|[![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Fearful_generated.jpg =257x "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Fearful_generated.wav)|


Here is the link to our [presentation](/EE599_final_presentation.pptx)<br/>

Here is a video showing a demo<br/>

Here is a link to our report<br/>

## Contributors <br/>
##### Shashank Hegde - [https://www.linkedin.com/in/karkala-shashank-hegde/](https://www.linkedin.com/in/karkala-shashank-hegde/)<br/>
##### Vineeth Ellore - [https://www.linkedin.com/in/vineethellore/](https://www.linkedin.com/in/vineethellore/) <br/>
##### Ashwin Telagimathada Ravi - [https://www.linkedin.com/in/ashwin-tr/](https://www.linkedin.com/in/ashwin-tr/)<br/>
