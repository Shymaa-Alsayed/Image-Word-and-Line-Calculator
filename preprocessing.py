import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import scipy.stats as stats
def process_image(img):
    #low = cv2.pyrDown(img)
    #lower = cv2.pyrDown(low)

    # blurred
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # thresholded
    ret, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh_img

def line_segmentation(img):
    lines_start_and_end=projection_histogram(img,type='vertical',dil=False)
    return lines_start_and_end

def crop_line(img,line_start_and_end):
    line_img = img[line_start_and_end[0]:line_start_and_end[1]]
    return line_img
def crop_word(img,word_start_and_end):
    word_img=img[:,word_start_and_end[0]:word_start_and_end[1]]
    return word_img

def word_segmentation(line_img,cutoff_value):
    no_words,line_img = calculate_no_words_in_line(line_img,cutoff_value)
    for i in range(1,16,2):
        for j in range(1,16,2):
            word_start_and_end = projection_histogram(line_img, type='horizontal',dil=True,window=(i,j))
            if (no_words==len(word_start_and_end))or (no_words==len(word_start_and_end)+1) or (no_words==len(word_start_and_end)-1):
                return word_start_and_end

def projection_histogram(img,type,dil,window=(3,3)):
    thresh=process_image(img)
    if type=='vertical':
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

    if type=='horizontal':
        if dil==True:
            kernel = np.ones(window, np.uint8)
            dilated = cv2.erode(img, kernel, iterations=1)  ##because ot is a white bckground, black letters
            ret, thresh = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        start_and_end=[]
        start=0
        end=0
        horizontal_histogram = thresh.sum(axis=0)
        for column in range(horizontal_histogram.size - 1):
            if ((horizontal_histogram[column] == 0) and (horizontal_histogram[column + 1] != 0)):
                start = column

            if (horizontal_histogram[column] != 0) and (horizontal_histogram[column + 1] == 0):
                end = column
                start_and_end.append((start, end))
        return start_and_end

def character_segmentation(img_name):
    characters=[]
    img=cv2.imread(img_name,0)
    lines_start_and_end=line_segmentation(img)
    print('lines')
    print(lines_start_and_end)
    for line in lines_start_and_end:
        line_img=crop_line(img,line)
        print('line')
        word_start_and_end = word_segmentation(line_img)
        print('words')
        print(word_start_and_end)
        for word in word_start_and_end:
            word_img=crop_word(line_img,word)
            cv2.imshow('word',word_img)
            cv2.waitKey()
            char_start_and_end=projection_histogram(word_img,type='horizontal',dil=False)
            print('chars')
            print(char_start_and_end)
            for char in char_start_and_end:
                char_img=crop_word(word_img,char)
                characters.append(char_img)

    cv2.destroyAllWindows()
    return characters

def calculate_spaces(line_img):
    thresh=process_image(line_img)
    projected_values=thresh.sum(axis=0)
    spaces=[]
    count = 1
    for i in range(1,len(projected_values)-1):
        if projected_values[i-1]==0:
           if  projected_values[i]==0:
               count+=1
           else:
               spaces.append(count)
               count=0
    return spaces


def calculate_no_words_in_line(line_img,cutoff_value):
    spaces=calculate_spaces(line_img)
    word_spaces=[ value for value in spaces if value>=cutoff_value]
    no_words=len(word_spaces)
    return  no_words,line_img



class Root(Tk):
    img_name=''
    def __init__(self):
        super(Root, self).__init__()
        self.title("Word and Line Calculator")
        self.minsize(500, 300)

        self.labelFrame = ttk.LabelFrame(self, text = "Open File")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)

        self.line_label=ttk.Label( text = "Number of lines is ")
        self.line_label.place(x = 20, y =155)

        self.word_label=ttk.Label( text = "Number of words is ")
        self.word_label.place(x = 20, y =175)

        self.avg_label=ttk.Label( text = "Average number of words in a line is ")
        self.avg_label.place(x = 20, y =195)

        self.browse_button()
        self.calculate_button()


    def browse_button(self):
        self.browse_button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.browse_button.grid(column = 1, row = 1)

    def calculate_button(self):
        self.calculate_button =ttk.Button(text='Calculate',command=self.run)
        self.calculate_button.place(x= 20, y=120)

    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")) )
        self.img_name=self.filename
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)

    def run(self):
        img = cv2.imread(self.img_name, 0)
        lines_coord = line_segmentation(img)
        total_no_of_lines = len(lines_coord)
        total_no_of_words = 0
        total_spaces=[]
        for line in lines_coord:
            line_img = crop_line(img, line)
            spaces=calculate_spaces(line_img)
            total_spaces.extend(spaces)
        loc, theta = stats.expon.fit(total_spaces)
        mean=np.ceil(theta)
        for line in lines_coord:
            line_img = crop_line(img, line)
            word_start_and_end = word_segmentation(line_img,mean)
            total_no_of_words += len(word_start_and_end)
        avg_no_words_in_line = round(total_no_of_words / len(lines_coord),1)

        self.line_label=ttk.Label( text = "")
        self.line_label.place(x = 125, y = 155)
        self.line_label.configure(text = total_no_of_lines)

        self.word_label=ttk.Label( text = "")
        self.word_label.place(x = 130, y = 175)
        self.word_label.configure(text = total_no_of_words)

        self.avg_label=ttk.Label( text = "")
        self.avg_label.place(x = 220, y = 195)
        self.avg_label.configure(text = avg_no_words_in_line)



root = Root()
root.mainloop()

