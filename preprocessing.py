import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import scipy.stats as stats
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt


def process_image(img):
    # low = cv2.pyrDown(img)
    # lower = cv2.pyrDown(low)

    # blurred
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # thresholded
    ret, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh_img


def line_segmentation(img):
    lines_start_and_end = projection_histogram(img, type='vertical', dil=False)
    return lines_start_and_end


def crop_line(img, line_start_and_end):
    line_img = img[line_start_and_end[0]:line_start_and_end[1]]
    return line_img


def crop_word(img, word_start_and_end):
    word_img = img[:, word_start_and_end[0]:word_start_and_end[1]]
    return word_img


def word_segmentation(line_img):
    no_words, line_img = calculate_no_words_in_line(line_img)
    word_start_and_end = []
    for i in range(1, 16, 2):
        for j in range(1, 16, 2):
            word_start_and_end = projection_histogram(line_img, type='horizontal', dil=True, window=(i, j))
            if (no_words == len(word_start_and_end)) or (no_words == len(word_start_and_end) + 1) or (
                    no_words == len(word_start_and_end) - 1):
                return word_start_and_end


def projection_histogram(img, type, dil, window=(3, 3)):
    thresh = process_image(img)
    if type == 'vertical':
        vertical_histogram = thresh.sum(axis=1)
        line_start_and_end_row = []
        start = 0
        end = 0
        for row in range(vertical_histogram.size - 1):
            if ((vertical_histogram[row] == 0) and (vertical_histogram[row + 1] != 0)):
                start = row
            if ((vertical_histogram[row] != 0) and (vertical_histogram[row + 1] == 0)):
                end = row
                line_start_and_end_row.append((start, end))
        return line_start_and_end_row

    if type == 'horizontal':
        if dil == True:
            kernel = np.ones(window, np.uint8)
            dilated = cv2.erode(img, kernel, iterations=1)  ##because ot is a white bckground, black letters
            ret, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        start_and_end = []
        start = 0
        end = 0
        horizontal_histogram = thresh.sum(axis=0)
        for column in range(horizontal_histogram.size - 1):
            if ((horizontal_histogram[column] == 0) and (horizontal_histogram[column + 1] != 0)):
                start = column

            if (horizontal_histogram[column] != 0) and (horizontal_histogram[column + 1] == 0):
                end = column
                start_and_end.append((start, end))
        return start_and_end


def character_segmentation(img_name):
    characters = []
    img = cv2.imread(img_name, 0)
    lines_start_and_end = line_segmentation(img)
    print('lines')
    print(lines_start_and_end)
    for line in lines_start_and_end:
        line_img = crop_line(img, line)
        print('line')
        word_start_and_end = word_segmentation(line_img)
        print('words')
        print(word_start_and_end)
        for word in word_start_and_end:
            word_img = crop_word(line_img, word)
            cv2.imshow('word', word_img)
            cv2.waitKey()
            char_start_and_end = projection_histogram(word_img, type='horizontal', dil=False)
            print('chars')
            print(char_start_and_end)
            for char in char_start_and_end:
                char_img = crop_word(word_img, char)
                characters.append(char_img)

    cv2.destroyAllWindows()
    return characters


def calculate_spaces(line_img):
    thresh = process_image(line_img)
    projected_values = thresh.sum(axis=0)
    spaces = []
    count = 1
    for i in range(1, len(projected_values) - 1):
        if projected_values[i - 1] == 0:
            if projected_values[i] == 0:
                count += 1
            else:
                spaces.append(count)
                count = 0
    return spaces


def calculate_no_words_in_line(line_img):
    spaces = calculate_spaces(line_img)
    mean_spaces=np.mean(spaces)
    #print('gamma mean', mean_spaces)
    word_spaces = [value for value in spaces if value > mean_spaces]
    no_words = len(word_spaces) - 1
    return no_words, line_img


def x_cor_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])

    else:
        cx = 0
    return cx


def square(im, desired_size):
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [1, 1, 1]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def vectorize_input(img):
    reshaped_img = np.reshape(img, (1, 32, 32, 1))
    return reshaped_img


def recognize_word(sorted_contours, thresh_word):
    word = []
    for c in sorted_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w > h:
            letter1 = thresh_word[y:y + h, x:x + int(w / 2)]
            # letter1=process_image(letter1)
            squared1 = square(letter1, 32)
            reshaped_img1 = vectorize_input(squared1)
            # rescaled1=cv2.resize(letter1,(32,32))
            # cv2.imshow('letter', squared1)
            search_value1 = model.predict_classes(reshaped_img1)[0]
            # print(model.predict_proba(reshaped_img))
            for key, value in classes.items():
                if value == search_value1:
                    word.append(key[0])
                    # print(key)
            # cv2.waitKey()
            letter2 = thresh_word[y:y + h, x + int(w / 2):x + w]
            # processed2=process_image(letter2)
            squared2 = square(letter2, 32)
            reshaped_img2 = vectorize_input(squared2)
            # rescaled2=cv2.resize(letter2,(32,32))
            # cv2.imshow('letter', squared2)
            search_value2 = model.predict_classes(reshaped_img2)[0]
            # print(model.predict_proba(reshaped_img))
            for key, value in classes.items():
                if value == search_value2:
                    word.append(key[0])
                    # print(key)
            # cv2.waitKey()
            continue
        else:
            letter = thresh_word[y:y + h, x:x + w]
            # letter=process_image(letter)
            squared = square(letter, 32)
            # rescaled=cv2.resize(letter,(32,32))
            reshaped_img = vectorize_input(squared)
            # cv2.imshow('letter', squared)
            search_value = model.predict_classes(reshaped_img)[0]
            # print(model.predict_proba(reshaped_img))
            for key, value in classes.items():
                if value == search_value:
                    word.append(key[0])
                    # print(key)
            # cv2.waitKey()

    # cv2.destroyAllWindows()
    return ''.join(word)


def recognize_line(line_coord, img):
    line = []
    line_img = crop_line(img, line_coord)
    word_start_and_end = word_segmentation(line_img)
    for word_coord in word_start_and_end:
        word_img = crop_word(line_img, word_coord)
        upimg = cv2.pyrUp(word_img)
        thresh_word = process_image(upimg)
        cv2.findContours(thresh_word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(thresh_word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=x_cor_contour, reverse=False)
        line.append(recognize_word(sorted_contours, thresh_word))
    return ' '.join(line)



def recognize_paragraph(img):
    paragraph = []
    lines_start_and_end = line_segmentation(img)
    for line_coord in lines_start_and_end:
        paragraph.append(recognize_line(line_coord, img))
    return '\n'.join(paragraph)


"""def recognize():
    img_name = ''
    img = cv2.imread(img_name, 0)
    lines_coord = line_segmentation(img)
    line_img = crop_line(img, lines_coord[0])
    word_start_and_end = word_segmentation(line_img)
    word_img = crop_word(img, word_start_and_end[0])"""


class Root(Tk):
    img_name = ''

    def __init__(self):
        super(Root, self).__init__()
        self.title("Word and Line Calculator")
        self.minsize(600, 500)

        self.labelFrame = ttk.LabelFrame(self, text="Open File")
        self.labelFrame.grid(column=0, row=1, padx=20, pady=20)

        self.line_label = ttk.Label(text="Number of lines is ")
        self.line_label.place(x=20, y=155)

        self.word_label = ttk.Label(text="Number of words is ")
        self.word_label.place(x=20, y=175)

        self.avg_label = ttk.Label(text="Average number of words in a line is ")
        self.avg_label.place(x=20, y=195)

        # self.textbox=ttk.Entry( width =40, justify=LEFT)
        self.textbox = Text(height=15, width=70)
        self.textbox.place(x=20, y=220)

        self.browse_button()
        self.calculate_button()

    def browse_button(self):
        self.browse_button = ttk.Button(self.labelFrame, text="Browse A File", command=self.fileDialog)
        self.browse_button.grid(column=1, row=1)

    def calculate_button(self):
        self.calculate_button = ttk.Button(text='Calculate', command=self.run)
        self.calculate_button.place(x=20, y=120)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.img_name = self.filename
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)

    def run(self):
        img = cv2.imread(self.img_name, 0)
        lines_coord = line_segmentation(img)
        total_no_of_lines = len(lines_coord)
        total_no_of_words = 0
        total_spaces = []
        for line in lines_coord:
            line_img = crop_line(img, line)
            word_start_and_end = word_segmentation(line_img)
            print(word_start_and_end)
            total_no_of_words += len(word_start_and_end)
        avg_no_words_in_line = round(total_no_of_words / len(lines_coord), 1)

        paragraph = recognize_paragraph(img)
        self.line_label = ttk.Label(text="")
        self.line_label.place(x=125, y=155)
        self.line_label.configure(text=total_no_of_lines)

        self.word_label = ttk.Label(text="")
        self.word_label.place(x=130, y=175)
        self.word_label.configure(text=total_no_of_words)

        self.avg_label = ttk.Label(text="")
        self.avg_label.place(x=220, y=195)
        self.avg_label.configure(text=avg_no_words_in_line)

        self.textbox.insert(END, paragraph)


model = load_model('cnn.h5')
classes_data = pickle.load(open('classes', 'rb'))
classes = classes_data['label_map']

root = Root()
root.mainloop()
