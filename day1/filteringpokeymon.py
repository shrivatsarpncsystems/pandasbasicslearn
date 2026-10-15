import pandas as pd


df = pd.read_csv("data.csv", index_col="Name")

# tall_pokemon = df[df["Height"] >= 2]
# print(tall_pokemon)

# legendRY_POKEMON = df[df["Legendary"] == True]
# print(legendRY_POKEMON)

# water_pokemon = df[(df["Type1"] == "Water") | (df["Type2"] == "Water")]
# print(water_pokemon)

# fierflying_pokemon = df[(df["Type1"] == "Fire") & (df["Type2"] == "Flying")]
# print(fierflying_pokemon)

