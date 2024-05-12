import numpy as np
from typing import Tuple, List
import pandas as pd

#=============================================================
    #HOW TO USE
#call the functions one after the other, 
    #the first one returns an array of the columns
    #the second one adds the arrays to the df as columns with unique names (just numbers)
#They didn't provide the tokens in the zip file, so I have no idea which words the frequencies are attached to

#a1_vec = keep_most_freq(cva1, 20) 
    #cva1 is a numpy file that was loaded like this:
        #cva1 = np.load("C:/Users/dengs/OneDrive/Documents/1ML/A2/project_data/project_data/features_countvec/features_countvec/train_countvec_features_actor_1_name.npy")
#add_vector_df(train_df, a1_vec)
    #train_df is the dataframe ur using
#=============================================================

def keep_most_freq(cv:np.ndarray, n:int)->List[List[int]]:
    nplen = len(cv[0])
    entirenplen = len(cv)
    thisnp = cv

    word_freq:List[int] = [0] * nplen
    for i in range(entirenplen):
        for x in range(nplen):
            word_freq[x] += thisnp[i][x]

    word_freq_sort = word_freq.copy()

    word_freq_sort.sort()

    set_word_freq = set(word_freq)

    num: List[int] = []

    for element in set_word_freq:
        num.append(element)

    num_freq: List[int] = [0] * len(set_word_freq)

    i = 0
    for element in word_freq_sort:
        if element != num[i]:
            i+=1
        num_freq[i] = num_freq[i] + 1
    #=========================
    #CREATE A LIST OF THE MOST FREQUENT APPEARANCES
    most_freq = word_freq_sort[len(word_freq_sort) - n - 1:-1]

    most_freq_i:List[int] = []

    for i in range(len(word_freq)):
        if word_freq[i] in most_freq:
            most_freq.remove(word_freq[i])
            most_freq_i.append(i)
    #=========================
    #NOW FOR EVERY VECTOR, ONLY KEEP THE NUMBERS AT THESE INDEXES

    new_all:List[List[int]] = []
    for i in range(len(cv)):
        new:List[int] = []
        for x in range(len(most_freq_i)):
            new.append(cv[i][most_freq_i[x]])
        new_all.append(new)
    #print(new_all)

    by_column:List[List[int]] = []
    for i in range(len(new_all[0])):
        #print(f"adding element {i} of column:")
        this:List[List[int]] = []
        for y in range(len(new_all)):
            #print(f"{y}")
            this.append(new_all[y][i])
        by_column.append(this)

    #print(by_column)
    return by_column

def add_vector_df(train_df:pd.DataFrame, columns:List[List[int]]):
    column_names:List[str] = []
    for i in range(len(columns[0])):
        column_names.append(str(i + 1))

    for i in range(len(columns)):
        train_df[column_names[i]] = columns[i]
    #display(train_df)    
