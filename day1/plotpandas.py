import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# df["Type1"].value_counts().plot(kind="bar")

type_count = df["Type1"].value_counts()
plt.barh(type_count.index,type_count.values)

plt.xlabel("Count")
plt.ylabel("Types")


plt.show()


