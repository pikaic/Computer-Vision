"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
from typing import List
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    enrollment(characters)

    list_image = detection(test_img)
    
    recognition(list_image)

    #raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    #img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    # Using SOBEL Filters for Edge Detection
    for i in range(0,5):    
        img = characters[i][1]
        sift=cv2.xfeatures2d.SIFT_create()
        kp,desc=sift.detectAndCompute(img,None)
        kp_name = "sift_keypoints_" + str(i) + ".txt"
        file1 = open(kp_name, 'w')
        if (str(kp)=='()' and str(desc) == 'None'):
            file1.write(str(img))
        else: 
            #print (np.size(desc,0),np.size(desc,1))
            for j in range(0,np.size(desc,0)):
                file1.write(str(desc[j]))
        file1.close()
    #print (characters[0][0], characters[1][0], characters[2][0], characters[3][0],characters[4][0])

    #raise NotImplementedError

def get_image(img,cc):

    row , col = img.shape
    rowflag = colflag = 0 # to know whether we have already found the start address or not 
    srow = scol = erow = ecol = 0
    image_list = []

    for k in range (0,cc):
        # find the start row & col 
        for i in range (0,row):
            for j in range (0,col):
                if rowflag == 0 and img[i][j] == cc:
                    srow = i
                    rowflag = 1 
                if colflag == 0 and img[i][j] == cc:
                    scol = j
                    colflag = 1
        # find the end row and col
        # find the end row
        for i in range (srow + 1,row): # image must be of atleast 1 pixel height
            for j in range (0,col):            
                if rowflag == 1:
                    if img[i][j] == cc:
                        erow = i
        # find the end col 
        for j in range(0,col):
            for i in range(0,row):
                    if img[i][j] == cc:
                        ecol = i                

        # now we need to store the image 
        cropped_image = img[srow:erow, scol:ecol]

        c_image = {
            "bbox" : cropped_image,
            "x" : srow,
            "y" : scol,
            "w" : (erow - srow),
            "h" : (ecol - scol),
            "name" : "NULL"
        }

        image_list.append(c_image)

        return image_list  



def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    # Connected Component Implementation / Naive Template Matching Implementation
    # Splitting of Test image into windows - 
    #print (np.size(test_img,0),np.size(test_img,1))
    # 
    row , col = test_img.shape
    
    #modifying the image into complete black & white image.
    th, test_img = cv2.threshold(test_img, 128, 255, cv2.THRESH_BINARY_INV)

    for i in range(0, row):
        for j in range(0, col):
            if test_img[i,j]==255:
                test_img[i,j]=1
            elif test_img[i,j]==0:
                test_img[i,j]=0

    
    cc = 0
    conflicts = 0  
    store_conflicts = [] # will do nested list here (basically a 2d list)
    for i in range (0,row):
        for j in range (0,col):
            if(test_img[i][j])>0:
                above=test_img[i-1][j]
                left=test_img[i][j-1]
            # INIT ABOVE, LEFT
            #finding out the above and the left elements
            # Normal condition
            # if i!=0 or j!=0:  
            #    above = test_img[i-1][j]
            #    left = test_img[i][j-1]
            # 1st element condition
            if above!=left:
                test_img[i][j]=min(above,left)
                
            elif i == 0 and j == 0:
                above = test_img[0][0]
                left = test_img[0][0]
            # 1st row  
            elif i == 0 and j!=0:
                left = test_img[i][j-1]
                above = 1500 
            # 1st col              
            elif i!= 0 and j == 0:
                above = test_img[i-1][j]
                left = 1500

            # above and left can't be zeros
            if above == 0:
                above = 1500
            elif left == 0:
                left = 1500
            
            # UPDATING LABELS
            # comparing above and left wrt to test_img[i][j]
            if test_img[i][j] != 0: # that means foreground element
                if conflicts == 0: 
                    if above == 0:
                        test_img[i][j] = min(cc + 1,left)
                    elif left == 0:
                        test_img[i][j] = min(cc + 1,above)
                    else:
                        test_img[i][j] = min(cc + 1,left,above)
                        if above != left: # above > left:
                            # conflict
                            conflicts = conflicts + 1
                            # all the pixels with value labelled as "above" will be relabelled as "left"
                            temp_list = []
                            temp_list.append(left)
                            temp_list.append(above)
                            store_conflicts.append(temp_list)
                            del temp_list
                #else:
                    
            # UPDATE CC
            if left != 0 and test_img[i][j] == 0: # transition from foreground to background  
                if left > cc:
                    cc = cc + 1
                # else:
                    # cc = cc + 1 
                    # cc will be same 
            #edge condition while updating
            if j == col and test_img[i][j] > cc:
                cc = cc + 1

    # Now modifying the image array wrt the conflict data
    # CONFLICT-REITERATION / RELABEL    
    for k in range (0,conflicts):
        for i in range (0,row):
            for j in range (0,col):
                if test_img[i][j] == store_conflicts[k][1]:
                    test_img[i][j] = store_conflicts[k][0]

    # Modified no. of connected components.
    cc = cc - conflicts
            
    image_list = get_image(test_img,cc)

    return image_list

    #raise NotImplementedError

def recognition():
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
