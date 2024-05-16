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
