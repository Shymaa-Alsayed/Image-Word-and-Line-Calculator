# Image Word and Line Calculator
Word and Line Calculator for any image format containing typed English text.
## Overview
* It is a Tkinter GUI App for functioning as a word and line calculator for images containing typed text of any size or font. It displays
three things: Total Number of lines in image - Total number of words in image - Average number of words in a line.
## Usage
run preprocessing.py to bring up the Tkinter app, select an image from your machine, click calculate to display the calculated values
## How it works
* Based on simple processing of the image and thresholding pixels to only zeros and ones, it uses a vertical projection histogram to count the
number of lines and their coordinates needed for line segmentation.
* For words, it uses two techniques together, first, it uses horizontal projection on each segmented line to count interspaces between words and characters and, second, it considers the spaces values as random variable modeling the width of spaces in a line, after that, it fits an exponential distribution to this data to get the mean space value, it considers any space above mean as an interword space and the rest are intercharacter spaces.
* Numer of words is the number of interword spaces-1. By adding up counted words in the segmented lines gives the total number of words and also the average of words number in a line.
