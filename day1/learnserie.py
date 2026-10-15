import pandas as pd 
# data = [1,2,3,4,5,6,7,8,9,10]
# serise = pd.Series(data, index=['a','b','c','d','e','f','g','h','i','j']) 
# print(serise)   
# print(serise.loc['a'])  # access by name
# print(serise.iloc[0]) # access by index
# print(serise[serise>=5])  



# a     1
# b     2
# c     3
# d     4
# e     5
# f     6
# g     7
# h     8
# i     9
# j    10
# dtype: int64



calories = {'day1': 2200, 'day2': 2300, 'day3': 2400, 'day4': 2500, 'day5': 2600}
series = pd.Series(calories)
# print(series)

# print(series.loc['day1'])
# series.loc['day1'] +=400
# print(series)







