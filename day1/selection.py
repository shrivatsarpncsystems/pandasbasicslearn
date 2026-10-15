import pandas as pd 

df = pd.read_csv("data.csv", index_col="Name")

# selection by coloumn using dot notation

# print(df.Name)
# print(df.Type1)

# selection by coloumn using bracket notation
# print(df["Name"])
# print(df["Type1"])
# print(df[["Name","Height","Weight"]]) use .toString to pring all

#selection by rows 
# print(df.iloc[0]) # first row
# print(df.iloc[0:5]) # first 5 rows

# print(df.loc["Blastoise":"Pikachu",["Height","Weight"]])

# print(df.iloc[0:11:2, 0:3])

# pokemon = input("Enter the pokemon name: ")
# try:
#     print(df.loc[pokemon])
# except KeyError:
#     print(f"{pokemon} not found!"ff)











