# Image Word and Line Calculator
Word and Line Calculator for any image format containing typed English text.
## Overview
* A Tkinter GUI App for functioning as  Optical Character Recognition and a word and line calculator for images containing typed English text on a  white background  of any size or font. It displays Four things: Total Number of lines in image - Total number of words in image - Average number of words in a line - Typed text of the image content.
## Usage
run preprocessing.py to bring up the Tkinter app, select an image from your machine, click calculate to display the results
## How it works
* Based on simple processing of the image and thresholding pixels to only zeros and ones, it uses a vertical projection histogram to count the
number of lines and their coordinates needed for line segmentation.
* For words, it uses two techniques together, first, it uses horizontal projection on each segmented line to count interspaces between words and characters and, second, it considers the spaces values as random variable modeling the width of spaces in a line, after that, it fits a gamma distribution to this data to get the mean space value, it considers any space above mean as an interword space and the rest are intercharacter spaces.
* Numer of words is the number of interword spaces-1. By adding up counted words in the segmented lines gives the total number of words and also the average of words number in a line.
* A trained CNN on the dataset :https://www.kaggle.com/passionoflife/english-typed-alphabets-and-numbers-dataset, is used to predict each character according to the segmentation strategy explained above.
