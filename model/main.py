import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle
def create_model(data):
    x = data.drop(['diagnosis'],axis=1)
    y = data['diagnosis']

    scalar = StandardScaler()
    x = scalar.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('Accuracy of model', accuracy_score(y_test,y_pred))
    print("classification Report :\n",classification_report(y_test,y_pred))
    return model, scalar

def get_cleaned_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})
    return data

def main():
    data = get_cleaned_data()

    model, scalar = create_model(data)

    with open('model/model.pkl', 'wb') as f :
        pickle.dump(model, f)
    with open('model/scalar.pkl','wb') as f:
        pickle.dump(scalar, f)   

if __name__ == '__main__':
    main()