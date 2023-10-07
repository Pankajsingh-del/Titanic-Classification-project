import numpy as nd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb

# load the data from csv file to pandas dataframe
titanic_data = pd.read_csv(r'C:\Users\user\Downloads\train.csv')

print(titanic_data)

# printing the first 5 rows odf the dataframe
titanic_data.head()
print(titanic_data.head())

# number of rows and columns
titanic_data.shape
print(titanic_data.shape)

# getting some information about the data
titanic_data.info()
print(titanic_data.info())

# check the no of missing values in each column
titanic_data.isnull().sum()
print(titanic_data.isnull().sum())

# drop the cabin column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin',axis= 1)
print (titanic_data)

# replacing the missing values in "age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace = True)
print(titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace = True))

# fiding the mode value of embarked column
print (titanic_data['Embarked'].mode())

print (titanic_data['Embarked'].mode()[0])

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace = True)

titanic_data.isnull().sum()
print(titanic_data.isnull().sum())

# getting ome statical aboutthe data
titanic_data.describe()
print(titanic_data.describe())

#fnding no of people of survived and not survived
titanic_data ['Survived'].value_counts()
print(titanic_data ['Survived'].value_counts())

titanic_data ['Sex'].value_counts()
print(titanic_data ['Sex'].value_counts())

#DATA VISUALISATION

sns.set()
# seperate the data into numeric and categorical
df_x = titanic_data[['Age','SibSp','Parch','Fare']]
df_y = titanic_data[['Survived','Pclass','Sex','Ticket','Embarked']]


for i in df_x.columns :
    plt.hist(df_x[i])
    plt.title(i)
    plt.show()
    print(plt.show)
    
    
#pivot table
    pd.pivot_table(titanic_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])
    print(pd.pivot_table(titanic_data, index = 'Survived', values = ['Age','SibSp','Parch','Fare']))
    
    #displot is a deprecated
   # sns.distplot(titanic_data['Age'][titanic_data['Survived']==0])
   # sns.distplot(titanic_data['Age'][titanic_data['Survived']==1])
   # print(sns.distplot(titanic_data['Age'][titanic_data['Survived']==0]))
   # print(sns.distplot(titanic_data['Age'][titanic_data['Survived']==1]))
    

#Encoding the categorial columns
titanic_data ['Sex'].value_counts()
print(titanic_data ['Sex'].value_counts())

titanic_data['Embarked'].value_counts()
print(titanic_data['Embarked'].value_counts())


# Data Preprocessing
titanic_data = titanic_data.drop(['Name', 'Ticket', 'Embarked'], axis=1)  # Remove unnecessary columns
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})  # Convert 'Sex' to numeric
titanic_data = titanic_data.dropna()  # Remove rows with missing values
X = titanic_data.drop('Survived', axis=1)  # Features
y = titanic_data['Survived']  # Target variable

titanic_data.head()
print(titanic_data.head())

titanic_data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

#separating features and target
X = titanic_data.drop(columns= ['Survived'],axis=1)
Y = titanic_data['Survived']

print(X)
print(Y)

#splitting the data into training data and test data
X_train , X_test, Y_train , Y_test = train_test_split (X,Y,test_size=0.2, random_state=12)
print(X.shape, X_train.shape, X_test.shape)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, Y_train)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test)

#Logistic Regression
model = LogisticRegression()

#training the logistic model with training data
male= float()
print(male)
model.fit(X_train,Y_train)

#ACCURACY SCORE
X_train_prediction = model.predict(X_train)

X_test_prediction = model.predict(X_test)

print(X_test_prediction)

traning_data_accuracy =accuracy_score(Y_train ,X_train_prediction)
print('Accuracy score of training data : ',traning_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)

print(X_test_prediction)

test_data_accuracy = accuracy_score(Y_test,X_test_prediction)
print('accuracy score of test data : ', test_data_accuracy)

