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

calm2anger - https://drive.google.com/uc?id=14q4aZseMCQO_xbbmX-JRsbRSlGX9bB3E <br />

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

|script|                                                      Calm                                                      |                                                         Angry                                                          |                                                         Fearful                                                          |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|"Dogs are sitting by the door"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_06.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_06.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_06_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_06_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_06_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_06_Fearful_generated.wav) |
|"Dogs are sitting by the door"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/GoodAudio/calm_04_Fearful_generated.wav) |

The above samples show that the spectrograms of angry and fearful speech have more predominant and spaced out harmonics. These are characteristics of angry and fearful speech.<br />


### Unseen audio from same data set:<br />

|script|                                                      Calm                                                      |                                                         Angry                                                          |                                                         Fearful                                                          |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|"Dogs are sitting by the door"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig1_Fearful_generated.wav) |
|"Kids are talking by the door"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_orig2_Fearful_generated.wav) |

<br />

### Same script by unseen actor:<br />

|script|                                                      Calm                                                      |                                                         Angry                                                          |                                                         Fearful                                                          |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|"The Dogs are sitting by the door"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng2_Fearful_generated.wav) |

### Lexically similar script by unseen actor:<br />

|script|                                                      Calm                                                      |                                                         Angry                                                          |                                                         Fearful                                                          |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|"This project is fun"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng1_Fearful_generated.wav) |
|"Three plus one equals four"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_eng3_Fearful_generated.wav) |

### Different laguage by unseen actor:<br />

|script|                                                      Calm                                                      |                                                         Angry                                                          |                                                         Fearful                                                          |
| :--------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|"Gaadi waala aya ghar se kachra nikal"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_hin_Fearful_generated.wav) |
|"Konegu project mugithu"|   [![Input Calm speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan.jpg "Input Calm speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan.wav)    |   [![Output Angry speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Anger_generated.jpg "Output Angry speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Anger_generated.wav) |   [![Output Fearful speech](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Fearful_generated.jpg "Output Fearful speech")](/GAN-for-speech-spectrogram/results/UnseenAudio/calm_kan_Fearful_generated.wav) |



Here is the link to our [presentation](/EE599_final_presentation.pptx)<br/>

Here is a video showing a demo<br/>

Here is a link to our report<br/>

## Contributors <br/>
##### Shashank Hegde - [https://www.linkedin.com/in/karkala-shashank-hegde/](https://www.linkedin.com/in/karkala-shashank-hegde/)<br/>
##### Vineeth Ellore - [https://www.linkedin.com/in/vineethellore/](https://www.linkedin.com/in/vineethellore/) <br/>
##### Ashwin Telagimathada Ravi - [https://www.linkedin.com/in/ashwin-tr/](https://www.linkedin.com/in/ashwin-tr/)<br/>
