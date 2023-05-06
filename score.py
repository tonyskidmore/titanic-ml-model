import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.joblib')

# print(train_df.head())
#    PassengerId  Survived  Pclass   Age  SibSp  Parch     Fare  Sex_female  Sex_male  Embarked_C  Embarked_Q  Embarked_S
# 0            1       0.0       3  22.0      1      0   7.2500       False      True       False       False        True
# 1            2       1.0       1  38.0      1      0  71.2833        True     False        True       False       False
# 2            3       1.0       3  26.0      0      0   7.9250        True     False       False       False        True
# 3            4       1.0       1  35.0      1      0  53.1000        True     False       False       False        True
# 4            5       0.0       3  35.0      0      0   8.0500       False      True       False       False        True

# Create a new dataframe with the same features as the training data
new_data = pd.DataFrame({
    'PassengerId': [999],
    'Pclass': [3],
    'Age': [25],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [10],
    'Sex_female': [1],
    'Sex_male': [0],
    'Embarked_C': [0],
    'Embarked_Q': [1],
    'Embarked_S': [0],
})

# Make predictions for the new data
predictions = model.predict(new_data)

if predictions[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger did not Survive")
