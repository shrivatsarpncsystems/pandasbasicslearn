import pandas as pd
import numpy as np

# Create a realistic messy employee dataset
data = {
    "Emp_ID": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 104, 111, 112, 113, 114, 115],
    "Name": [
        "  Alice Smith ", "BOB JONES", "Charlie Brown", "David Lee", 
        "Eva Green", " Frank White ", "Grace Kelly", "Hannah Abbott", 
        "Ian Malcolm", "Jack Sparrow", "David Lee", "Karen Page", 
        "Leo Messi", "Mia Wallace", "Nathan Drake", "Olivia Pope"
    ],
    "Age": [
        "28", "34", "unknown", "42", "29", "250", "31", "24", 
        "45", "38", "42", "29", "36", "N/A", "33", "41"
    ],
    "Department": [
        "IT", "it ", "HR", "Marketing", "Finance", "IT", "hr", 
        np.nan, "Sales", "IT", "Marketing", "Finance", "Sales", "HR", "Sales", "Marketing"
    ],
    "Salary": [
        "$75,000", "$68,000", "$52,000", "$95,000", "82000", 
        "$110,000", "N/A", "$48,000", "$90,000", "$105,000", 
        "$95,000", "$78,000", "$130,000", "$60,000", "88,000", "$92,000"
    ],
    "Experience_Years": [3.5, 7.0, 1.0, 12.0, 4.0, 15.0, np.nan, 0.5, 14.0, 10.0, 12.0, 5.0, 13.0, 2.0, 8.5, 11.0],
    "Rating": [4.2, 3.8, 2.9, 4.9, 4.0, 1.5, 3.5, 4.1, 4.7, 3.9, 4.9, 3.7, 4.8, 3.0, 4.4, 4.6],
    "Remote_Worker": ["Yes", "No", "yes", "NO", "Yes", "no", "Yes", "No", "Yes", "yes", "NO", "No", "Yes", "No", "Yes", "No"],
    "Left_Company": [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]  # Target for ML (Churn/Attrition)
}

df_messy = pd.DataFrame(data)
df_messy.to_csv("day1/messy_employee_data.csv", index=False)
print("-> Created day1/messy_employee_data.csv")

# Create a clean department budget Excel file to teach reading Excel and merging
dept_data = {
    "Department": ["IT", "HR", "Marketing", "Finance", "Sales"],
    "Dept_Head": ["Sarah Connor", "Jim Halpert", "Don Draper", "Gordon Gekko", "Jordan Belfort"],
    "Budget_Million": [2.5, 0.8, 1.4, 1.9, 3.2]
}
df_dept = pd.DataFrame(dept_data)
df_dept.to_excel("day1/department_budget.xlsx", index=False)
print("-> Created day1/department_budget.xlsx")
