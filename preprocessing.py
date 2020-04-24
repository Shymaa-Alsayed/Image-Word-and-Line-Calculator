import cv2
import numpy as np

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

def word_segmentation(line_img):
    word_start_and_end = projection_histogram(line_img, type='horizontal',dil=True)
    return  word_start_and_end

def projection_histogram(img,type,dil):
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
            kernel = np.ones((5, 5), np.uint8)
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
               count=0#
    return spaces

def run(img_name):
    img=cv2.imread(img_name,0)
    lines_coord=line_segmentation(img)
    total_spaces=[]
    for i in range(len(lines_coord)):
        line_img=crop_line(img,lines_coord[i])
        spaces=calculate_spaces(line_img)
        total_spaces.extend(spaces)
    mean_space=np.mean(total_spaces)
    std_space=np.std(total_spaces)
    no_words_plus_one = [x for x in total_spaces if x > mean_space + std_space]
    no_words = len(no_words_plus_one) - 1
    return  no_words,total_spaces

img_name='btest.PNG'
characters=character_segmentation(img_name)
