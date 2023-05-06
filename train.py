import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier(n_estimators=100, random_state=42)


# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Combine the train and test data
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Remove unnecessary columns
# 'PassengerId',
# , 'Cabin'
combined_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values
combined_df['Age'].fillna(combined_df['Age'].mean(), inplace=True)
combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0], inplace=True)
combined_df['Fare'].fillna(combined_df['Fare'].mean(), inplace=True)

# Convert categorical variables to numerical variables
combined_df = pd.get_dummies(combined_df, columns=['Sex', 'Embarked'])

# Split the combined data back into train and test sets
train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df):]

print(train_df.head())

# Split the training set into training and validation sets
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print('Best hyperparameters:', grid_search.best_params_)

model = grid_search.best_estimator_

y_pred = model.predict(X_val)

print(classification_report(y_val, y_pred))
print('AUC-ROC score:', roc_auc_score(y_val, y_pred))

X_test = test_df.drop(['Survived'], axis=1)
y_pred = model.predict(X_test)

# convert Survived column to int or else submission will result in 0 score
output = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': y_pred.astype(int)})
output.to_csv('submission.csv', index=False)

joblib.dump(model, 'model.joblib')

# Load the model from a file
# model = joblib.load('model.joblib')

# https://www.kaggle.com/competitions/titanic/leaderboard?search=skidmore
