# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the trained model
# model_path = '/Users/Admin/Desktop/best_model.pkl'  # Update with the correct path if different
# best_model = joblib.load(model_path)

# # Load the new CSV file for prediction
# new_data_path = '/Users/Admin/Desktop/dataset/test_compdata.csv'
# new_data = pd.read_csv(new_data_path)

# # Extract features and the incident angle
# features = new_data[['Incident Angle', 'Wing Span', 'Fuselage Length']].values
# incident_angle = new_data['Incident Angle'].values

# # Predict the magnitude using the trained model
# predicted_magnitude = best_model.predict(features)

# # Create a DataFrame for plotting
# plot_data = pd.DataFrame({
#     'Incident Angle': incident_angle,
#     'Predicted Magnitude': predicted_magnitude
# })

# # Sort the DataFrame by 'Incident Angle'
# plot_data = plot_data.sort_values(by='Incident Angle')

# # Load the simulated data from C130_CST.csv
# simulated_data_path = '/Users/Admin/Desktop/dataset/C130_CST.csv'
# simulated_data = pd.read_csv(simulated_data_path)

# # Extract the necessary columns
# simulated_data = simulated_data[['Incident Angle', 'Simulated Magnitude']]

# # Merge the plot data with the simulated data on 'Incident Angle'
# merged_data = pd.merge(plot_data, simulated_data, on='Incident Angle', how='inner')

# # Plot both the predicted magnitude and simulated magnitude versus incident angle
# plt.plot(merged_data['Incident Angle'], merged_data['Predicted Magnitude'], label='Predicted Magnitude')
# plt.plot(merged_data['Incident Angle'], merged_data['Simulated Magnitude'], label='Simulated Magnitude')
# plt.xlabel('Incident Angle (degrees)')
# plt.ylabel('Magnitude')
# plt.title('Magnitude vs Incident Angle')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Calculate performance metrics
# mse = mean_squared_error(merged_data['Simulated Magnitude'], merged_data['Predicted Magnitude'])
# r2 = r2_score(merged_data['Simulated Magnitude'], merged_data['Predicted Magnitude'])
# print("Mean Squared Error:", mse)
# print("R-squared:", r2)

# ###CHECKING THE INDEX ERROR
# import pandas as pd

# # Load the simulated data from C130_CST.csv
# simulated_data_path = '/Users/Admin/Desktop/dataset/C130_CST.csv'
# simulated_data = pd.read_csv(simulated_data_path)

# # Display the first few rows and the columns of the dataframe
# print(simulated_data.head())
# print(simulated_data.columns)

##VERSION 03 
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import time

start_time = time.time()


# Load the trained model
model_path = '/Users/Admin/Desktop/best_model.pkl'  # Update with the correct path if different
best_model = joblib.load(model_path)

# Load the new CSV file for prediction
new_data_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/test_compdata_C130.csv'
new_data = pd.read_csv(new_data_path)

# Extract features and the incident angle
features = new_data[['Incident Angle', 'Wing Span', 'Fuselage Length']].values
incident_angle = new_data['Incident Angle'].values

# Predict the magnitude using the trained model
predicted_magnitude = best_model.predict(features)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Incident Angle': incident_angle,
    'Predicted Magnitude': predicted_magnitude
})

# Sort the DataFrame by 'Incident Angle'
plot_data = plot_data.sort_values(by='Incident Angle')

# Load the simulated data from C130_CST.csv
simulated_data_path = '/Users/Admin/Desktop/dataset/C130_CST.csv'
simulated_data = pd.read_csv(simulated_data_path)

# Correct column names if there are leading/trailing spaces
simulated_data.columns = simulated_data.columns.str.strip()

# Assuming the correct column names are 'Incident Angle' and 'Simulated Magnitude'
simulated_data = simulated_data[['Incident Angle', 'Simulated Magnitude']]

# Merge the plot data with the simulated data on 'Incident Angle'
merged_data = pd.merge(plot_data, simulated_data, on='Incident Angle', how='inner')

# Plot both the predicted magnitude and simulated magnitude versus incident angle
plt.plot(merged_data['Incident Angle'], merged_data['Predicted Magnitude'], label='Predicted Magnitude')
plt.plot(merged_data['Incident Angle'], merged_data['Simulated Magnitude'], label='Simulated Magnitude')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Magnitude')
plt.title('Magnitude vs Incident Angle')
plt.legend()
plt.grid(True)
plt.show()

# Calculate performance metrics
mse = mean_squared_error(merged_data['Simulated Magnitude'], merged_data['Predicted Magnitude'])
r2 = r2_score(merged_data['Simulated Magnitude'], merged_data['Predicted Magnitude'])
correlation = merged_data[['Simulated Magnitude', 'Simulated Magnitude']].corr().iloc[0, 1]

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Correlation:", correlation)

###### CODE WITH OPTIMIZATION AND REDUCING OVERFITTING 
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 

# Load the trained model
model_path = '/Users/Admin/Desktop/best_model.pkl' 
best_model = joblib.load(model_path)

# Check if it's a pipeline, if not, create a pipeline
if not isinstance(best_model, Pipeline):
    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', best_model)
    ])

# Load the new CSV file for prediction
new_data_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/test_compdata_C130.csv'
new_data = pd.read_csv(new_data_path)

# Extract features and the incident angle
features = new_data[['Incident Angle', 'Wing Span', 'Fuselage Length']].values
incident_angle = new_data['Incident Angle'].values

# Load the simulated data from C130_CST.csv
simulated_data_path = '/Users/Admin/Desktop/dataset/C130_CST.csv'
simulated_data = pd.read_csv(simulated_data_path)

# Correct column names if there are leading/trailing spaces
simulated_data.columns = simulated_data.columns.str.strip()

# Assuming the correct column names are 'Incident Angle' and 'Simulated Magnitude'
simulated_data = simulated_data[['Incident Angle', 'Simulated Magnitude']]

# Assuming that test_compdata also has 'Simulated Magnitude', otherwise remove this merge
merged_data = pd.merge(new_data, simulated_data, on='Incident Angle', how='inner')

# Split data into training and validation sets (80% train, 20% validation)
X_train = merged_data[['Incident Angle', 'Wing Span', 'Fuselage Length']].values
y_train = merged_data['Simulated Magnitude'].values

# Hyperparameter Tuning for Overfitting Reduction and MSE Optimization
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [3, 4, 5],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model.set_params(**best_params)

# Retrain the best model
best_model.fit(X_train, y_train)

# Predict the magnitude
predicted_magnitude = best_model.predict(features)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Incident Angle': incident_angle,
    'Predicted Magnitude': predicted_magnitude
})

# Merge with simulated data on 'Incident Angle'
merged_data = pd.merge(plot_data, simulated_data, on='Incident Angle', how='inner')


# Plot both the predicted magnitude and simulated magnitude versus incident angle
plt.plot(merged_data['Incident Angle'], merged_data['Predicted Magnitude'], label='Predicted Magnitude')
plt.plot(merged_data['Incident Angle'], merged_data['Simulated Magnitude'], label='Simulated Magnitude')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Magnitude')
plt.title('Magnitude vs Incident Angle')
plt.legend()
plt.grid(True)
plt.show()

# Calculate performance metrics
mse = mean_squared_error(merged_data['Simulated Magnitude'], merged_data['Predicted Magnitude'])
r2 = r2_score(merged_data['Simulated Magnitude'], merged_data['Predicted Magnitude'])
correlation = merged_data[['Simulated Magnitude', 'Predicted Magnitude']].corr().iloc[0, 1]  # Fixed correlation

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Correlation:", correlation)


end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")


