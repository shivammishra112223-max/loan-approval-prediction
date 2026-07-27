import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
Data=pd.read_csv(r"C:\Users\shiva\Downloads\archive (9)\loan_approval_dataset.csv")
Data.columns=Data.columns.str.strip()
print(Data.shape)

print(Data.isnull().sum())

Data['education']=Data['education'].fillna(Data['education'].mode()[0])
Data['self_employed']=Data['self_employed'].fillna(Data['self_employed'].mode()[0])
print(Data.isnull().sum())



print(Data.columns)
print(Data.info())
# chek outlier 

columns=["no_of_dependents","income_annum","loan_amount","loan_term","cibil_score","residential_assets_value","commercial_assets_value","luxury_assets_value","bank_asset_value"]
for col in columns:
    Q1=Data[col].quantile(0.25)
    print(col)
    Q3=Data[col].quantile(0.75)
    
    IQR=Q3 - Q1
    
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    
    outlier= Data[(Data[col] < lower) | (Data[col] > upper)]
    print(f"{col} : {len(outlier)} Outliers")



columns = [
    "residential_assets_value",
    "commercial_assets_value",
    "bank_asset_value"
]

for col in columns:

    Q1 = Data[col].quantile(0.25)
    Q3 = Data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_before = Data[(Data[col] < lower) | (Data[col] > upper)]
    print(f"Before Clipping -> {col}: {len(outlier_before)} Outliers")

    Data[col] = Data[col].clip(lower=lower, upper=upper)

    outlier_after = Data[(Data[col] < lower) | (Data[col] > upper)]
    print(f"After Clipping -> {col}: {len(outlier_after)} Outliers\n")

#Dublicate data

print(Data.duplicated().sum(),"duplicate")    

#Data types chek

print(Data.dtypes)

#Encoding 
#print(Data.head(20))
Data['education'] = Data['education'].str.strip()
Data['self_employed'] = Data['self_employed'].str.strip()
Data['loan_status'] = Data['loan_status'].str.strip()

Data['education'] = Data['education'].map({'Graduate':1,'Not Graduate':0})
Data['self_employed'] = Data['self_employed'].map({'Yes':1,'No':0})
Data['loan_status'] = Data['loan_status'].map({'Approved':1,'Rejected':0})
#Scaling start




scaler=MinMaxScaler()
Data[['no_of_dependents','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']]=scaler.fit_transform(
    Data[['no_of_dependents','income_annum','loan_amount','loan_term','cibil_score','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']]
)




#Feature & splite 

X = Data.drop(columns=['loan_id', 'loan_status'])
Y = Data['loan_status']
# train-test


print(X.isnull().sum())
X_train,X_test,Y_train,Y_test=train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model=RandomForestClassifier(
   
)
model.fit(X_train,Y_train)
joblib.dump(model, "loan_model.pkl")
prediction=model.predict(X_test)
print(prediction)



print("accuracy",accuracy_score(Y_test,prediction))

print("precision",precision_score(Y_test,prediction))
print("recall",recall_score(Y_test,prediction))
print("F-1 ",f1_score(Y_test,prediction))
print("confusion matrics",confusion_matrix(Y_test, prediction))

# save
joblib.dump(scaler, "scaler.pkl")
joblib.dump(scaler, "scaler.pkl")




































#New customer

# # ---------------- New Customer ----------------

# import pandas as pd
# import joblib

# # Model Load
# model = joblib.load("loan_model.pkl")

# # Scaler Load
# scaler = joblib.load("scaler.pkl")

# new_customer = pd.DataFrame({
#     "no_of_dependents":[2],
#     "education":[1],
#     "self_employed":[0],
#     "income_annum":[1200000],
#     "loan_amount":[5000000],
#     "loan_term":[15],
#     "cibil_score":[780],
#     "residential_assets_value":[3000000],
#     "commercial_assets_value":[1000000],
#     "luxury_assets_value":[2000000],
#     "bank_asset_value":[1500000]
# })

# # जिन Columns पर Training में Scaling की थी
# scale_columns = [
#     'no_of_dependents',
#     'income_annum',
#     'loan_amount',
#     'loan_term',
#     'cibil_score',
#     'residential_assets_value',
#     'commercial_assets_value',
#     'luxury_assets_value',
#     'bank_asset_value'
# ]

# # Scaling
# new_customer[scale_columns] = scaler.transform(
#     new_customer[scale_columns]
# )

# # Prediction
# prediction = model.predict(new_customer)

# if prediction[0] == 1:
#     print("Loan Approved")
# else:
#     print("Loan Rejected")