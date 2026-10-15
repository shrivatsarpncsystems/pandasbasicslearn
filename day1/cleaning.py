import pandas as pd 

df = pd.read_csv("data.csv")


# df = df.drop(columns=["Legendary","No"]) drop specified columns

# handel missing data

# df = df.dropna(subset=["Type2"])  # drop missing values

# df = df.fillna({"Type2":"None"}) # missing values change name

# drop duplicate values

# df = df.drop_duplicates()

#fix inconsistent values 

# df["Type1"] = df["Type1"].replace({"Fire":"Fighting"}) 


# print(df.to_string())

# df["Legendary"] = df["Legendary"].astype(bool) 


# print(df.to_string())



