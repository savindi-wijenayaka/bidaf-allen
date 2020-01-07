# BiDAF-AllenNLP Edited
This is a Edited version of AllenNLP BiDAF implementation for Machine Comprehension Tasks. In this I separated the Tokenization and Embedding part of the passage. In the first calling pass the passage and a sample question. After that you can just ask the question. This is coded just for experience and research.

## BiDAF Model Architecture
![BiDaF_Arch](https://github.com/PhantomGrin/bidaf-allen/blob/master/images/bidaf_architecture.png)

(Source: https://allenai.github.io/bi-att-flow/)

## Tokenization and Instance Creation -> Path

## Making the Tensors -> Path

## Issues

* Time Taken for answering a question based on a large passage
https://github.com/allenai/allennlp/issues/2571

<!-- 
```Changeable Thresholds:
emotion_weight = 0.6 	//values between 0 to 1
pose_weight = 0.4	//values between 0 to 1

emotion_marks = | emo_positive - total_sentiment | * emotion_weight
guesture_marks = gues_positive * pose_weight

overall_marks = emotion_marks + guesture_marks
``` -->

## DISCLAIMER
As per Apache License 2.0 (Apache-2.0) of AllenNLP, they are not liable to any of this modifications I made. 
