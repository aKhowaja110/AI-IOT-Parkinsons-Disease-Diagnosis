from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd
import pickle
import xgboost as xgb

df = pd.read_csv("datasets/Training-Final.csv")
X=df.drop(['LABEL'],axis=1)
scaler=StandardScaler()
X=scaler.fit_transform(X)
y=df['LABEL']
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(x_train, y_train)
y_pred=xgb_model.predict(x_test)
print("XGBoost accuracy",accuracy_score(y_test,y_pred))

weights = xgb_model.feature_importances_
with open("models/tremor__model_weights.txt", "w") as output:
    output.write(str(weights))

pickle.dump(xgb_model, open('models/tremor_model.sav', 'wb'))

