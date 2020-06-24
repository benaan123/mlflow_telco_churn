import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from utils.tenure_lab import tenure_lab

def feature_engineering(telcom):
	"""
	Takes in dataframe from file and performs feature engineering.
	Outputs dataframe with features, scaled dataframe, columns for modelling and target column.

	"""

	#Replacing spaces with null values in total charges column
	telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)

	#Dropping null values from total charges column which contain .15% missing data 
	telcom = telcom[telcom["TotalCharges"].notnull()]
	telcom = telcom.reset_index()[telcom.columns]

	#convert to float type
	telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

	#replace 'No internet service' to No for the following columns
	replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies']

	for i in replace_cols:
		telcom[i]  = telcom[i].replace({'No internet service' : 'No'})

	#replace values
	telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes",0:"No"})

	telcom["tenure_group"] = telcom.apply(lambda telcom:tenure_lab(telcom), axis = 1)

	#Separating churn and non churn customers
	churn     = telcom[telcom["Churn"] == "Yes"]
	not_churn = telcom[telcom["Churn"] == "No"]

	#Separating catagorical and numerical columns
	Id_col     = ['customerID']
	target_col = ["Churn"]
	cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
	cat_cols   = [x for x in cat_cols if x not in target_col]
	num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

	#customer id col
	Id_col     = ['customerID']
	#Target columns
	target_col = ["Churn"]
	#categorical columns
	cat_cols   = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
	cat_cols   = [x for x in cat_cols if x not in target_col]
	#numerical columns
	num_cols   = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]
	#Binary columns with 2 values
	bin_cols   = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
	#Columns more than 2 values
	multi_cols = [i for i in cat_cols if i not in bin_cols]

	#Label encoding Binary columns
	le = LabelEncoder()
	for i in bin_cols :
		telcom[i] = le.fit_transform(telcom[i])

	#Duplicating columns for multi value columns
	telcom = pd.get_dummies(data = telcom,columns = multi_cols )

	#Scaling Numerical columns
	std = StandardScaler()
	scaled = std.fit_transform(telcom[num_cols])
	scaled = pd.DataFrame(scaled,columns=num_cols)

	return [telcom, scaled, Id_col, target_col]
