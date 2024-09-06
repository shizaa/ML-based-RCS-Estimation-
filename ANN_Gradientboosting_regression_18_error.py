##VERSION 01
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import time

# start_time = time.time()

# # Load the CSV file
# data = pd.read_csv('/Users/Admin/Desktop/dataset/compdata.csv')

# # Split the data into features and target
# X = data.iloc[:, :-1].values  # Features
# y = data.iloc[:, -1].values   # Target

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define the pipeline including feature scaling and model
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('regressor', GradientBoostingRegressor())
# ])

# # Define the parameter grid for hyperparameter tuning with fewer combinations
# param_grid = {
#     'regressor__n_estimators': [50, 100, 200],
#     'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'regressor__max_depth': [3, 4, 5],
#     'regressor__subsample': [0.8, 0.9, 1.0],
#     'regressor__min_samples_split': [2, 5, 10],
#     'regressor__min_samples_leaf': [1, 2, 4],
#     'regressor__max_features': ['sqrt', 'log2']
# }

# # Perform grid search to find the best parameters
# grid_search = GridSearchCV(estimator=pipeline,
#                            param_grid=param_grid,
#                            scoring='neg_mean_squared_error',
#                            cv=5,
#                            n_jobs=-1,
#                            verbose=2)

# # Train the model
# grid_search.fit(X_train, y_train)

# # Get the best model
# best_model = grid_search.best_estimator_

# # Make predictions on the test set
# y_pred = best_model.predict(X_test)


# # Visualize predicted vs. actual labels
# plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
# plt.scatter(range(len(y_pred)), y_pred, color='r', label='Predicted')
# plt.xlabel('Sample Index')
# plt.ylabel('Magnitude')
# plt.title('Gradient Boosting Regression')
# plt.legend()
# plt.show()

# # Add predicted values to the DataFrame
# data['predicted_magnitude'] = best_model.predict(data.iloc[:, :-1].values)

# # Save the updated DataFrame to a new CSV file
# data.to_csv('/Users/Admin/Desktop/nn_compdata_predicted.csv', index=False)

# # Calculate performance metrics
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print("Mean Squared Error:", mse)
# print("R-squared:", r2)

# end_time = time.time()
# execution_time = end_time - start_time

# print("Execution time:", execution_time, "seconds")

####### VERSION 02 WHICH WE ARE GOING TO USE FOR FYP THESIS DEFENSE PRESENTATION
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import time 

start_time = time.time()

# Load the CSV file
data = pd.read_csv('/Users/Admin/Desktop/dataset/BGR_training/compdata.csv')

# Split the data into features and target
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline including feature scaling and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', GradientBoostingRegressor())
])

# Define the parameter grid for hyperparameter tuning with fewer combinations
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [3, 4, 5],
    'regressor__subsample': [0.8, 0.9, 1.0],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2']
}

# Perform grid search to find the best parameters
grid_search = GridSearchCV(estimator=pipeline,
                            param_grid=param_grid,
                            scoring='neg_mean_squared_error',
                            cv=5,
                            n_jobs=-1,
                            verbose=0)

# Train the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Save the best model to a file
joblib.dump(best_model, '/Users/Admin/Desktop/best_model.pkl')



# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Visualize predicted vs. actual labels
plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='r', label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Gradient Boosting Regression')
plt.legend()
plt.show()

# Add predicted values to the DataFrame
data['predicted_magnitude'] = best_model.predict(data.iloc[:, :-1].values)

# Save the updated DataFrame to a new CSV file
data.to_csv('/Users/Admin/Desktop/dataset/BGR_training/nn_compdata_predicted.csv', index=False)



# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)



end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")



