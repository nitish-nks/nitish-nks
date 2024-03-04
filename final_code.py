#######################################
# Importing required librarie Pipeline#
#######################################

print("Step 1: Required librarie imported successfully")


import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB









####################
# To ignore warning#
####################

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)





################################################
# Loading online_shoppers_intention.csv dataset#
################################################

print("Step 2: Created DataFrame successfully")

df = pd.read_csv("online_shoppers_intention.csv")












######################
# Feature Engineering#
######################

print("Step 3: Feature Engineering Done successfully on Weekend, Revenue")


df['Weekend'] = df['Weekend'].replace((True, False), (1, 0))
df['Revenue'] = df['Revenue'].replace((True, False), (1, 0))

condition = df['VisitorType']=='Returning_Visitor'





#################################
# Added Returning_Visitor column#
#################################

print("Step 4: Added Returning_Visitor column successfully")

df['Returning_Visitor'] = np.where(condition, 1, 0)

df = df.drop(columns=['VisitorType'])











############################################
# Applying One Hot Encoding on Month column#
############################################

print("Step 5: Applied one hot encoding successfully on Month column")


ordinal_encoder = OrdinalEncoder()
df['Month'] = ordinal_encoder.fit_transform(df[['Month']])










#########################################
# Checking correlation on Revenue column#
#########################################

print("Step 6: Checking correlation done successfully")


result = df[df.columns[1:]].corr()['Revenue']						
result1 = result.sort_values(ascending=False)









###########################################
# Prepairing Features as X and target as y#
###########################################

print("Step 7: Prepairing features as X and target as y done successfully")


X = df.drop(['Revenue'], axis=1)
y = df['Revenue']











####################################
# Prepairing Train and Test Dataset#
####################################

print("Step 8: Splitting data X_train, X_test, y_train & y_test done successfully")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)














#################
# Model Pipeline#
#################

print("Step 9: model_pipeline fcuntion created done successfully")


def model_pipeline(X, model):  
    n_c = X.select_dtypes(exclude=['object']).columns.values.tolist()
    c_c = X.select_dtypes(include=['object']).columns.values.tolist()

    numeric_columns = n_c
    categorical_columns = c_c

    numeric_pipeline = SimpleImputer(strategy = 'constant')

    categorical_pipeline = OneHotEncoder(handle_unknown = 'ignore')

    a = ('numeric', numeric_pipeline, numeric_columns)
    b = ('categorical', categorical_pipeline, categorical_columns)

    preprocessor = ColumnTransformer(

    transformers = [a, b], 
    remainder = 'passthrough'

    )

    c = ('preprocessor', preprocessor)
    d = ('smote', SMOTE(random_state = 1))
    e = ('scaler', MinMaxScaler())
    f = ('feature_selection', SelectKBest(score_func = chi2, k = 6))
    g = ('model', model)

    bundled_pipeline = imbpipeline(steps = [c, d, e, f, g])

    return bundled_pipeline







##################
# Model Selection#
##################


print("Step 10: select_model fcuntion created done successfully")


def select_model(X, y, pipeline=None):

    classifiers = {}
    

    c_d4 = {"RandomForestClassifier": RandomForestClassifier()}
    classifiers.update(c_d4)

    c_d5 = {"DecisionTreeClassifier": DecisionTreeClassifier()}
    classifiers.update(c_d5)

    c_d9 = {"KNeighborsClassifier": KNeighborsClassifier()}
    classifiers.update(c_d9)

    c_d10 = {"RidgeClassifier": RidgeClassifier()}
    classifiers.update(c_d10)

    c_d13 = {"BernoulliNB": BernoulliNB()}
    classifiers.update(c_d13)

    c_d14 = {"SVC": SVC()}
    classifiers.update(c_d14)
   
    cols = ['model', 'run_time', 'roc_auc']
    df_models = pd.DataFrame(columns = cols)

    for key in classifiers:
        
        start_time = time.time()
        
        print()
        print("Step 12: model_pipeline run successfully on", key)

        pipeline = model_pipeline(X_train, classifiers[key])
        
        cv = cross_val_score(pipeline, X, y, cv=10, scoring='roc_auc')

        row = {'model': key,
               'run_time': format(round((time.time() - start_time)/60,2)),
               'roc_auc': cv.mean(),
        }

        df_models = pd.concat([df_models, pd.DataFrame([row])], ignore_index=True)
        
    df_models = df_models.sort_values(by='roc_auc', ascending=False)
	
    return df_models
    


#####################################
# Access Model select_model function#
#####################################

print("Step 11: Accessing select_model function done successfully")


models = select_model(X_train, y_train)




###################################
# Lets see total model with score #
###################################

print("Step 13: Accessing select_model function done successfully")

print(models)





#####################################
# Accessing best model and training #
#####################################

print("Step 14: Accessing select_model function done successfully")

selected_model = SVC()
bundled_pipeline = model_pipeline(X_train, selected_model)
bundled_pipeline.fit(X_train, y_train)






#####################################
# Accessing best model and training #
#####################################

print("Step 15: Results predicted successfully")
y_pred = bundled_pipeline.predict(X_test)

print(y_pred)








#####################
# ROC and AOC score #
#####################

print("Step 16: ROC and AOC scores")

roc_auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)


print('ROC/AUC:', roc_auc)
print('Accuracy:', accuracy)
print('F1 score:', f1_score)









#########################
# Classification report #
#########################

print("Step 17: classification report generated successfully")

classif_report = classification_report(y_test, y_pred)

print(classif_report)



########################################
# BOSS its a right time to celebrate :)#
########################################


