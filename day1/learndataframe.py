import pandas as pd 

data = {"Name":['Alice','Bob','Charlie','David'], "Age":[25,30,35,40]}
dataframe = pd.DataFrame(data,index=["a","b","c","d"])
# print(dataframe.loc[["a","c"]])

# add a new coloumn name to dataframe 
dataframe["City"]=["New York","Los Angeles","Chicago","Houston"]
# print(dataframe)

# add a new row name to dataframe 
# dataframe.loc["e"] = ["Frank", 45, "Los Angeles"]
# new_row_data = pd.DataFrame([{"Name": "Frank", "Age": 45, "City": "Los Angeles"}], index = ["e"])
# df = pd.concat([dataframe,new_row_data])



# print(df)








# df = pd.DataFrame(data)
# print(df.T)
# print(df)
