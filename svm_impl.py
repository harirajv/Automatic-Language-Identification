import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train_path='FILE_NAME.csv'#mention file name or path
train_data=pd.read_csv(train_path)
y=train_data.Lang
feat=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','F30','F31','F32','F33','F34','F35','F36','F37','F38','F39']
X=train_data[feat]
model=SVC(kernel='rbf')
train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2)
model.fit(train_X,train_y)
val_predictions=model.predict(val_X)
print(accuracy_score(val_y,val_predictions))
