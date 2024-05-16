import numpy as np
from typing import Tuple, List, Union
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import math
import matplotlib.pyplot as plt

from typing import List

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn import preprocessing

#================================================================================
def add_engineered_dir_features(df:pd.DataFrame):
    unique_dir = df["director_name"].unique()
    dir_movie_count: List[int] = [0] * len(df) #some random huge number. jut in case each director appears only once in the dataset

    nmovies_for_director: List[int] = [1] * len(df)
    dir_over_mean: List[int] = [0] * len(df)
    dir_over_std1: List[int] = [0] * len(df)

    print(f"there are {len(unique_dir)} udirs")
    for udir in unique_dir:
        count: int = 0
        for dir in df["director_name"]:
            if dir == udir:
                count+=1
        print(f"udir: {udir} has {count} appearances")

        dir_movie_count[count] += 1

        if count > 1:
            print(" movies:")
            for index, row in df[df["director_name"] == udir].iterrows():
                nmovies_for_director[index] = count

    nmovie_mean = np.mean(nmovies_for_director)
    nmovie_std1 = nmovie_mean + np.std(nmovies_for_director)

    for count in nmovies_for_director:
        #add it to a further array for 'well-known directors'
        if count >= nmovie_mean:
            dir_over_mean[index] = 1

        if count >= nmovie_mean + nmovie_std1:
            dir_over_std1[index] = 1

    df["nmovies_director"] = nmovies_for_director
    df["dir_over_mean"] = dir_over_mean
    df["dir_over_std1"] = dir_over_std1

#===========================================================================================

def get_dir_avgross(df:pd.DataFrame):
    average_gross = ((df.groupby(["director_name"]))["gross"]).mean()

    avgross_dir:List[int] = [0] * len(df)

    for index, row in df.iterrows():
        avgross_dir[index] = average_gross[row["director_name"]]

    df["avgross_dir"] = avgross_dir

#===========================================================================================

def sort_func(arr:List[List[Union[str,int]]]):
    return arr[1]


def n_top_countries(df:pd.DataFrame, n:int):
    unique_cnt = df["country"].unique()

    #print(f"there are {len(unique_cnt)} udirs")

    n_cnts: List[List[Union[str,int]]] = []

    for ucnt in unique_cnt:
        count: int = 0
        ucnt_arr = [ucnt, 0]
        for cnt in df["country"]:
            if cnt == ucnt:
                ucnt_arr[1]+=1
        #print(f"udir: {ucnt} has {ucnt_arr[1]} appearances")
        n_cnts.append(ucnt_arr)

    #print(n_cnts)

    cnt_sorted = sorted(n_cnts, key=sort_func)

    cnt_sorted = cnt_sorted[-n:]
    #print(cnt_sorted)

    #now create columns for these top n countries
    for cnt in cnt_sorted:
        this_cnt:List[int] = []
        for row in df["country"]:
            #print(row)
            if row == cnt[0]:
                #print("YES")
                this_cnt.append(1)
            else:
                this_cnt.append(0)
        df[cnt[0]] = this_cnt

#============================================================================================

def n_top_languages(df:pd.DataFrame, n:int):
    unique_lan = df["language"].unique()

    #print(f"there are {len(unique_lan)} udirs")

    n_lans: List[List[Union[str,int]]] = []

    for ulan in unique_lan:
        count: int = 0
        ulan_arr = [ulan, 0]
        for lan in df["language"]:
            if lan == ulan:
                ulan_arr[1]+=1
        print(f"udir: {ulan} has {ulan_arr[1]} appearances")
        n_lans.append(ulan_arr)

    #print(n_lans)

    lan_sorted = sorted(n_lans, key=sort_func)

    lan_sorted = lan_sorted[-n:]
    print(lan_sorted)

    #now create columns for these top n countries
    for lan in lan_sorted:
        this_lan:List[int] = []
        for row in df["language"]:
            print(row)
            print(f"trying to match with {lan[0]}")
            if row == lan[0]:
                
                print("YES")
                this_lan.append(1)
            else:
                this_lan.append(0)
        df[lan[0]] = this_lan
