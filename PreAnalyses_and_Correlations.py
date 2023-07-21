# Python script, pre analysis, correlation
# Data Analysis
# First, assess correlation

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

df_raw = pd.read_csv('alldata.csv')
df_raw.columns
df_parameters = df_raw.drop(columns=['Order', 'ID.1', 'DistanceToUrbanArea', 'Order_x', 'COLLECTING DATE',
       'COLLECTING month', 'COLLECTING day', 'COLLECTING year', 'LATITIUDE',
       'LONGITUDE_x', 'BillAndHead', 'Species', 'Sex', 'Glucose_1', 'Glucose_2', 'Ketones_1', 'Ketones_2', 'UAZ NUMBER', 'SPECIES',
       'SPECIES_SYNONYM', 'COMMON NAME', 'AGE', 'SEX', 'Unnamed: 22',
       'COLLECTOR', 'PREPARER', 'COLLECTOR NUMBER', 'COUNTRY', 'STATE',
       'COUNTY', 'LOCATION', 'SPECIMEN PREPARATION', 'NOTES',
       'Exact coordinates?', 'MULTIPLE RESULTS', 'Time',
       'Molt; Miscellaneous; Softparts', 'BandNumber', 'ColorCombo',
       'Last of the Wild Value'])
corr_matrix = df_parameters.corr()

corr_matrix.to_csv('correlationmatrix.csv')



df_adults = df_raw.loc[df_raw["AGE"] != "Immature"]


df_NOCA_M = df_raw.loc[df_raw["Species"] == "NOCA"].loc[df_raw["Sex"] == "M"]
df_NOCA_F = df_raw.loc[df_raw["Species"] == "NOCA"].loc[df_raw["Sex"] == "F"]
df_PYRR_M = df_raw.loc[df_raw["Species"] == "PYRR"].loc[df_raw["Sex"] == "M"]
df_PYRR_F = df_raw.loc[df_raw["Species"] == "PYRR"].loc[df_raw["Sex"] == "F"]


# NOCA M correlations
df_parameters = df_NOCA_M.drop(columns=['Order', 'ID.1', 'DistanceToUrbanArea', 'Order_x', 'COLLECTING DATE',
       'COLLECTING month', 'COLLECTING day', 'COLLECTING_year', 'LATITIUDE',
       'LONGITUDE_x', 'BillAndHead', 'Species', 'Sex', 'Glucose_1', 'Glucose_2', 'Ketones_1', 'Ketones_2', 'UAZ NUMBER', 'SPECIES',
       'SPECIES_SYNONYM', 'COMMON NAME', 'AGE', 'SEX', 'Unnamed: 22',
       'COLLECTOR', 'PREPARER', 'COLLECTOR NUMBER', 'COUNTRY', 'STATE',
       'COUNTY', 'LOCATION', 'SPECIMEN PREPARATION', 'NOTES',
       'Exact coordinates?', 'MULTIPLE RESULTS', 'Time',
       'Molt; Miscellaneous; Softparts', 'BandNumber', 'ColorCombo',
       'Last of the Wild Value'])

corr_matrix = df_parameters.corr()
corr_matrix.to_csv('correlationmatrix_NOCA_M.csv')




# NOCA F correlations
df_parameters = df_NOCA_F.drop(columns=['Order', 'ID.1', 'DistanceToUrbanArea', 'Order_x', 'COLLECTING DATE',
       'COLLECTING month', 'COLLECTING day', 'COLLECTING_year', 'LATITIUDE',
       'LONGITUDE_x', 'BillAndHead', 'Species', 'Sex', 'Mass', 'CloacalWidth', 'CloacalDepth', 'Glucose_avg', 'Ketones_avg', 'Glucose_1', 'Glucose_2', 'Ketones_1', 'Ketones_2', 'UAZ NUMBER', 'SPECIES',
       'SPECIES_SYNONYM', 'COMMON NAME', 'AGE', 'SEX', 'Unnamed: 22',
       'COLLECTOR', 'PREPARER', 'COLLECTOR NUMBER', 'COUNTRY', 'STATE',
       'COUNTY', 'LOCATION', 'SPECIMEN PREPARATION', 'NOTES',
       'Exact coordinates?', 'MULTIPLE RESULTS', 'Time',
       'Molt; Miscellaneous; Softparts', 'BandNumber', 'ColorCombo',
       'Last of the Wild Value'])

corr_matrix = df_parameters.corr()
corr_matrix.to_csv('correlationmatrix_NOCA_F.csv')


# PYRR M correlations
df_parameters = df_PYRR_M.drop(columns=['Order', 'ID.1', 'DistanceToUrbanArea', 'Order_x', 'COLLECTING DATE',
       'COLLECTING month', 'COLLECTING day', 'COLLECTING_year', 'LATITIUDE',
       'LONGITUDE_x', 'BillAndHead', 'Species', 'Sex', 'Glucose_1', 'Glucose_2', 'Ketones_1', 'Ketones_2', 'UAZ NUMBER', 'SPECIES',
       'SPECIES_SYNONYM', 'COMMON NAME', 'AGE', 'SEX', 'Unnamed: 22',
       'COLLECTOR', 'PREPARER', 'COLLECTOR NUMBER', 'COUNTRY', 'STATE',
       'COUNTY', 'LOCATION', 'SPECIMEN PREPARATION', 'NOTES',
       'Exact coordinates?', 'MULTIPLE RESULTS', 'Time',
       'Molt; Miscellaneous; Softparts', 'BandNumber', 'ColorCombo',
       'Last of the Wild Value'])

corr_matrix = df_parameters.corr()
corr_matrix.to_csv('correlationmatrix_PYRR_M.csv')



# PYRR F correlations
df_parameters = df_PYRR_F.drop(columns=['Order', 'ID.1', 'DistanceToUrbanArea', 'Order_x', 'COLLECTING DATE',
       'COLLECTING month', 'COLLECTING day', 'COLLECTING_year', 'LATITIUDE',
       'LONGITUDE_x', 'BillAndHead', 'Species', 'Sex', 'Glucose_1', 'Glucose_2', 'Ketones_1', 'Ketones_2', 'UAZ NUMBER', 'SPECIES',
       'SPECIES_SYNONYM', 'COMMON NAME', 'AGE', 'SEX', 'Unnamed: 22',
       'COLLECTOR', 'PREPARER', 'COLLECTOR NUMBER', 'COUNTRY', 'STATE',
       'COUNTY', 'LOCATION', 'SPECIMEN PREPARATION', 'NOTES',
       'Exact coordinates?', 'MULTIPLE RESULTS', 'Time',
       'Molt; Miscellaneous; Softparts', 'BandNumber', 'ColorCombo',
       'Last of the Wild Value'])

corr_matrix = df_parameters.corr()
corr_matrix.to_csv('correlationmatrix_PYRR_F.csv')







# Then, run PCA on correlated variables

For all species and sexes, only H_breast_diff and S_breast_diff (-0.833008191)

# NOCA_M
Bill length and glucose are correlated (0.906748813)
Glucose and ketones are correlated (-0.845680819)
H_crest_diff and S_crest_diff are correlated (-0.758034744)
H_breast_diff and S_breast_diff are correlated (-0.81091784)

# NOCA_F
Perfect correlation between 'CloacalWidth', 'CloacalDepth', 'Glucose_avg', 'Ketones_avg', but only 2 individuals with data (dropping them all)
Mass is correlated with a lot but only has 3 individuals sampled (dropping it)

Head and BillLength are correlated (-0.771984368)
B_bill_diff and Wing are correlated (-0.777263704)
S_crest_diff and B_bill_diff are correlated (0.823921725)
H_face_diff and Tarsus are correlated (0.756538807)
H_face_diff and H_bill_diff are correlated (0.77013199)
H_breast_diff and S_breast_diff are correlated (-0.925259113)

# PYRR_M
CloacalDepth and Wing are correlated (0.905653038)









# Finally, linear model
df = df_adults.drop(columns=['Order', 'Order_x', 'COLLECTING DATE',
       'COLLECTING month', 'COLLECTING day', 'LATITIUDE',
       'LONGITUDE_x', 'BillAndHead', 'Glucose_1', 'Glucose_2', 'Ketones_1', 'Ketones_2', 'UAZ NUMBER', 'SPECIES',
       'SPECIES_SYNONYM', 'COMMON NAME', 'AGE', 'SEX', 'Unnamed: 22',
       'COLLECTOR', 'PREPARER', 'COLLECTOR NUMBER', 'COUNTRY', 'STATE',
       'COUNTY', 'LOCATION', 'SPECIMEN PREPARATION', 'NOTES',
       'Exact coordinates?', 'MULTIPLE RESULTS', 'Time',
       'Molt; Miscellaneous; Softparts', 'BandNumber', 'ColorCombo',
       'Last of the Wild Value'])


df.columns
Index(['ID', 'Source', 'DistanceToUrbanArea', 'COLLECTING_year', 'Species', 'Sex',
       'Mass', 'TailLength', 'Crest', 'Tarsus', 'Wing', 'BillLength', 'Head',
       'BillWidth', 'CloacalWidth', 'CloacalDepth', 'Glucose_avg',
       'Ketones_avg', 'H_bill_diff', 'S_bill_diff', 'B_bill_diff',
       'H_crest_diff', 'S_crest_diff', 'B_crest_diff', 'H_face_diff',
       'S_face_diff', 'B_face_diff', 'H_breast_diff', 'S_breast_diff',
       'B_breast_diff', 'vit. a', 'vio', 'anther', 'lutein', 'zeaxan',
       'b_crypto', 'b_dunno'],
      dtype='object')

df.to_csv('df_lm.csv')


import pandas as pd
import scipy
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.linear_model import LogisticRegression


model = smf.ols(formula='DistanceToUrbanArea ~ COLLECTING_year, Species, Sex, Mass, TailLength, Crest, Tarsus, Wing, BillLength, Head, BillWidth, CloacalWidth, CloacalDepth, Glucose_avg, Ketones_avg, H_bill_diff, S_bill_diff, B_bill_diff,  H_crest_diff, S_crest_diff, B_crest_diff, H_face_diff, S_face_diff, B_face_diff, H_breast_diff, S_breast_diff, B_breast_diff', data=df).fit()

df = df.dropna()
model = smf.ols(formula='DistanceToUrbanArea ~ Crest, Tarsus, Wing', data=df).fit()

y, X = patsy.dmatrices(f'pre_{i} ~ Course + LGBTQ + Gender + Religion', data=df)
X_df.as.data.frame(X)
backwardSelection(X, y)
