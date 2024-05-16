def parse_average(arr: str) -> float:
    #output_array: List[float] = []
    arr = arr[1:-1]
    array: List[str] = arr.split()
    total: float = 0
    for i in range(len(array)):
        total = total + float(array[i])
    return total / len(array)

def average_title_embedding(train_df: pd.DataFrame):
    print(len(train_df["title_embedding"]))
    av: pd.Series = train_df["title_embedding"].apply(parse_average)
    train_df["average_title_embedding"] = av  

knn_train = pd.read_csv("C:/Users/dengs/OneDrive/Documents/1ML/A2/project_data/project_data/train_dataset.csv")

#======================================================================

column_types = knn_train.dtypes

i:int = 0
remove:List[str] = []
string_feature:List[str] = []
for column in knn_train:
    if column_types[column] == "object":
        print(f"{column} is string")
        remove.append(column)
        if(column != "title_embedding"):
            string_feature.append(column)
        continue
    else:
        corr = knn_train[column].corr(knn_train["imdb_score_binned"])
        print(f"    {column} has corr: {corr}")
        if abs(corr) < 0.3:
            remove.append(column)
    i+=1

#===============================================================================
average_title_embedding(knn_train)
for column in string_feature:
    knn_train = knn_train.drop(column, axis=1)

knn_train = knn_train.drop("title_embedding", axis=1);
knn_train = knn_train.drop("id", axis=1)

for col in remove:
    if (col != 'id') and (col not in string_feature) and (col != 'title_embedding'):
        knn_train = knn_train.drop(col, axis=1)

knn_X = knn_train.drop("imdb_score_binned", axis='columns')
knn_y = knn_train["imdb_score_binned"]


knn_X = pd.DataFrame(normalize(knn_X,norm="l1", axis=1), columns=knn_X.columns, index=knn_X.index)



#=============================================================================
#   CREATING FOLDS

skf = StratifiedKFold()
n_splits:int = 5
x_neighbour: List[int] = []
for i in range(0,199):
    x_neighbour.append(i)

print(x_neighbour)

y_accuracy: List[float] = []

d_metric = "euclidean"
w_metric = "distance"

max_accuracy = 0
max_n = 0

for n in range(1,200):
    av_accuracy: float = 0
    knn = KNeighborsClassifier(n_neighbors=n, metric=d_metric, weights=w_metric)
    for train, test in skf.split(knn_X, knn_y):

        #print(type(train))
    

        #train = (x,y)
        #test = (x,y)
        x_train, x_test = knn_X.iloc[train], knn_X.iloc[test]
        y_train, y_test = knn_y.iloc[train], knn_y.iloc[test]

        knn.fit(x_train, y_train)

        y_pred = knn.predict(x_test)

        accuracy = accuracy_score(y_test,y_pred)
        av_accuracy += accuracy

        print(f"n = {n}. accuracy = {accuracy}")

    av_accuracy = av_accuracy / n_splits
    print(av_accuracy)
    y_accuracy.append(av_accuracy)
    if av_accuracy > max_accuracy:
        max_accuracy = av_accuracy
        max_n = n

print(y_accuracy)
print(f"max accuracy: {max_accuracy} for n: {max_n}")

#=====================================================================
# PLOT, show for specific neighbour number n

plt.scatter(x_neighbour, y_accuracy)
plt.xlabel("K Neighbours")
plt.ylabel("Accuracy")
plt.title(f"Scatterplot of K neighbours Against their Accuracy, ({d_metric}, normalised, weighted)")
plt.grid(True)
plt.show()

av_accuracy: float = 0
n = 47
#max = 0
knn = KNeighborsClassifier(n_neighbors=n, metric=d_metric, weights='distance')
for train, test in skf.split(knn_X, knn_y):
    #i is fold number
    print(f"train: {train}. test: {test}")

    #print(type(train))
    
    x_train, x_test = knn_X.iloc[train], knn_X.iloc[test]
    y_train, y_test = knn_y.iloc[train], knn_y.iloc[test]

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    av_accuracy += accuracy

    print(f"n = {n}. accuracy = {accuracy}")

av_accuracy = av_accuracy / n_splits
print(av_accuracy)
