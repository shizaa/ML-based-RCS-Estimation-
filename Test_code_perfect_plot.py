##VERSION 1
# #### TESTING 
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib

# # Load the trained model
# model_path = '/Users/Admin/Desktop/best_model.pkl'
# best_model = joblib.load(model_path)

# # Load the new CSV file
# new_data_path = '/Users/Admin/Desktop/dataset/test_compdata.csv'
# new_data = pd.read_csv(new_data_path)

# # Extract features and the incident angle
# features = new_data[['Incident Angle', 'Wing Span', 'Fuselage Length']].values
# incident_angle = new_data['Incident Angle'].values

# # Predict the magnitude using the trained model
# predicted_magnitude = best_model.predict(features)

# # Plot the predicted magnitude versus incident angle
# plt.plot(incident_angle, predicted_magnitude, label='Predicted Magnitude')
# plt.xlabel('Incident Angle (degrees)')
# plt.ylabel('Magnitude')
# plt.title('Magnitude vs Incident Angle')
# plt.legend()
# plt.grid(True)
# plt.show()

##VERSION 2
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib

# # Load the trained model
# model_path = '/Users/Admin/Desktop/best_model.pkl' 
# best_model = joblib.load(model_path)

# # Load the new CSV file
# new_data_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/C17_comptestdata.csv'
# new_data = pd.read_csv(new_data_path)

# # Extract features and the incident angle
# features = new_data[['Incident Angle', 'Wing Span', 'Fuselage Length']].values
# incident_angle = new_data['Incident Angle'].values


# # Predict the magnitude using the trained model
# predicted_magnitude = best_model.predict(features)


# # Save the results to a CSV file
# save_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/testC17_predicted_magnitudes.csv' 
# pd.DataFrame({'Predicted Magnitude': predicted_magnitude}).to_csv(save_path, index=False)


# # Create a DataFrame for plotting
# plot_data = pd.DataFrame({
#     'Incident Angle': incident_angle,
#     'Predicted Magnitude': predicted_magnitude
# })

# # Sort the DataFrame by 'Incident Angle'
# plot_data = plot_data.sort_values(by='Incident Angle')

# # Plot the predicted magnitude versus incident angle
# plt.plot(plot_data['Incident Angle'], plot_data['Predicted Magnitude'], label='Predicted Magnitude')
# plt.xlabel('Incident Angle (degrees)')
# plt.ylabel('Magnitude')
# plt.title('Magnitude vs Incident Angle')
# plt.legend()
# plt.grid(True)
# plt.show()

##############################3##VERSION 3
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

start_time = time.time()


# Load the trained model
model_path = '/Users/Admin/Desktop/best_model.pkl'
best_model = joblib.load(model_path)

# Load the new CSV file
new_data_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/test_compdata_C130.csv'
new_data = pd.read_csv(new_data_path)

# Extract features and the incident angle
features = new_data[['Incident Angle', 'Wing Span', 'Fuselage Length']].values
incident_angle = new_data['Incident Angle'].values

# Predict the magnitude using the trained model
predicted_magnitude = best_model.predict(features)

# Create a DataFrame for both features and predicted magnitudes
combined_data = pd.DataFrame({
    'Incident Angle': new_data['Incident Angle'],   # Include original incident angle
    'Wing Span': new_data['Wing Span'],
    'Fuselage Length': new_data['Fuselage Length'],
    'Predicted Magnitude': predicted_magnitude
})

# Save the results to a CSV file
save_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/test_compdata_C130.csv'
combined_data.to_csv(save_path, index=False)

# Create a DataFrame for plotting (no changes here)
plot_data = pd.DataFrame({
    'Incident Angle': incident_angle,
    'Predicted Magnitude': predicted_magnitude
})

# Sort the DataFrame by 'Incident Angle'
plot_data = plot_data.sort_values(by='Incident Angle')

# Plot the predicted magnitude versus incident angle (no changes here)
plt.plot(plot_data['Incident Angle'], plot_data['Predicted Magnitude'], label='Predicted Magnitude')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Magnitude')
plt.title('Magnitude vs Incident Angle')
plt.legend()
plt.grid(True)
plt.show()


end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")


##################################
######################################check for flate plate
#VERSION 01
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.metrics import mean_squared_error, r2_score

# # Load the trained model
# model_path = '/Users/Admin/Desktop/best_model.pkl'  
# best_model = joblib.load(model_path)

# # Load the new CSV file for prediction
# new_data_path = '/Users/Admin/Desktop/TestDataFlatPlate.csv'
# new_data = pd.read_csv(new_data_path)

# # Extract features and the incident angle
# features = new_data[['Length', 'Height', 'Incident Angle']].values
# incident_angle = new_data['Incident Angle'].values

# # Predict the magnitude using the trained model
# predicted_magnitude = best_model.predict(features)

# # Create a DataFrame for plotting
# plot_data = pd.DataFrame({
#     'Incident Angle': incident_angle,
#     'Predicted Magnitude': predicted_magnitude
# })

# # Sort the DataFrame by 'Incident Angle' (optional for better visualization)
# plot_data = plot_data.sort_values(by='Incident Angle')

# # Plot the predicted magnitude versus incident angle
# plt.figure(figsize=(10, 6))  # Adjust figure size if needed
# plt.plot(plot_data['Incident Angle'], plot_data['Predicted Magnitude'], marker='o', linestyle='-', color='blue')  # Added marker and line style
# plt.xlabel('Incident Angle (degrees)', fontsize=12)
# plt.ylabel('Predicted Magnitude', fontsize=12)
# plt.title('Predicted Magnitude vs Incident Angle', fontsize=14)
# plt.grid(True)
# plt.show()



#VRSION 2
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# import numpy as np

# # Load the best model
# best_model = joblib.load('/Users/Admin/Desktop/best_model.pkl')

# # Load the test data
# test_data = pd.read_csv('/Users/Admin/Desktop/TestDataFlatPlate.csv')

# # Ensure column names match (case-insensitive)
# test_data.columns = [col.lower() for col in test_data.columns]

# # Check and convert data types
# for col in ['length', 'height', 'incident angle']:
#     if test_data[col].dtype != 'float64':  # Check if float
#         test_data[col] = pd.to_numeric(test_data[col], errors='coerce')  # Convert to numeric, force errors to NaN

# # Handle missing values
# test_data.dropna(inplace=True)

# # Scaling
# scaler = joblib.load('/Users/Admin/Desktop/scaler.pkl')  # Load scaler if it was used
# test_data[['length', 'height', 'incident angle']] = scaler.transform(test_data[['length', 'height', 'incident angle']]) 

# # Creating custom input data
# custom_input_data = pd.DataFrame({
#     'length': test_data['length'].iloc[0],
#     'height': test_data['height'].iloc[0],
#     'incident angle': np.arange(-100, 101, 10)
# })

# # Make predictions
# custom_input_data['Predicted Magnitude'] = best_model.predict(custom_input_data[['length', 'height', 'incident angle']])

# # Plotting with refined aesthetics and axis limits
# plt.figure(figsize=(12, 8))  
# plt.plot(custom_input_data['incident angle'], custom_input_data['Predicted Magnitude'], marker='o', linestyle='-', color='blue', linewidth=2)
# plt.title('Predicted Magnitude vs. Incident Angle', fontsize=16)
# plt.xlabel('Incident Angle (degrees)', fontsize=14)
# plt.ylabel('Predicted Magnitude', fontsize=14)
# plt.xticks(np.arange(-100, 101, 20), fontsize=12)
# plt.yticks(np.arange(-180, 181, 20), fontsize=12)
# plt.grid(True, linestyle='--') # Optional: Add gridlines
# plt.xlim([-100, 100])
# plt.ylim([-180, 180])

# plt.tight_layout()
# plt.show()

# # Save the updated test data with predictions
# custom_input_data.to_csv('/Users/Admin/Desktop/Updated_TestDataFlatPlate.csv', index=False)

# print("Prediction, plotting, and data saving completed.")


