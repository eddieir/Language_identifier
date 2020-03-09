
The goal of this program is to use the knowledge of n-grams to implement a language identification tool. Currently it supports 5 languages: English, Malay, Dutch, Swedish, Spanish The reason for these languages is because of familiarity for EN and MS, DE and SV has similar language structure (phonemes, letters used etc), and ES because it's widely used in the USA. I did noticed there are some issues of underfitting if not enough training are given and bigram is used.

The current specifications of this program is by using Trigram and Quadgram as a combination to get the highest match between the trained models and the test set (string). Each language are trained with around 250000 words, resulting the number of unique tri- and quad- grams of around 4500 - 7000.

**Note: This project is defined based on Python3**

In order to run this project after clone in favor of train the language model it need to use **python3 language_word_identifier.py train -i training** and training sets will be save in the training folder
Once the train is done we could make predictions with using of **python3 language_word_identifier.py predict**

Example output:
![Screenshot from 2020-03-09 15-33-16](https://user-images.githubusercontent.com/23243761/76222845-71bbe600-621b-11ea-932b-4b5fd5dda6f8.png)

