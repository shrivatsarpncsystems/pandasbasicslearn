import pandas as pd

df = pd.read_csv("data.csv")

# pokemon_groupby = df.groupby("Type1")
# print(pokemon_groupby["Height"].max())
# print(pokemon_groupby["Height"].min())
# print(pokemon_groupby["Height"].mean())
# print(pokemon_groupby["Height"].sum())
# print(pokemon_groupby["Height"].count())
# print(pokemon_groupby["Height"].median())
# print(pokemon_groupby["Height"].mode())
# print(pokemon_groupby["Height"].std())

# print(df.groupby("Type1")["Height"].agg(["min","max","mean","sum","count","median","std"]))
# print(df.mean(numeric_only=True))   
# print(df.median(numeric_only=True)) 
# print(df.mode(numeric_only=True)) 
# print(df.std(numeric_only=True))    
# print(df.min(numeric_only=True))    
# print(df.max(numeric_only=True))    
# print(df.sum(numeric_only=True))    
# print(df.count(numeric_only=True))   
 

