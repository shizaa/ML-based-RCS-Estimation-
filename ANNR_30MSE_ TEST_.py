import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, schedules



# Load the saved Keras model
model_path = '/Users/Admin/Desktop/best_ann_model.h5' 
best_model = tf.keras.models.load_model(model_path)


# Load the new CSV file (replace with your actual path)
new_data_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/test_compdata_C130.csv'
new_data = pd.read_csv(new_data_path)

# Assuming the new data has the same structure as the original training data
features = new_data.iloc[:, :-1].values # All columns except the last (magnitude)
incident_angle = new_data['Incident Angle'].values

# Load the trained scaler
scaler_filename = '/Users/Admin/Desktop/all shizz/scaler.pkl'  # File to save the scaler
scaler = joblib.load(scaler_filename) # Load the saved scaler
features = scaler.transform(features) # Scale the features using loaded scaler


# Predict the magnitude
predicted_magnitude = best_model.predict(features).flatten()

# Create a DataFrame for both features and predicted magnitudes
combined_data = pd.DataFrame({
    'Incident Angle': incident_angle,
    'Wing Span': new_data['Wing Span'],
    'Fuselage Length': new_data['Fuselage Length'],
    'Predicted Magnitude': predicted_magnitude
})

# Save the results to a CSV file (replace with your desired path)
save_path = '/Users/Admin/Desktop/dataset/TEST_COMPFILES_PERFECT_PLOTS/ANNR_test_compdata_C130_predicted.csv'
combined_data.to_csv(save_path, index=False)

# Plotting
plot_data = pd.DataFrame({
    'Incident Angle': incident_angle,
    'Predicted Magnitude': predicted_magnitude
})
plot_data = plot_data.sort_values(by='Incident Angle')

plt.plot(plot_data['Incident Angle'], plot_data['Predicted Magnitude'], label='Predicted Magnitude', marker='o')
plt.xlabel('Incident Angle (degrees)')
plt.ylabel('Magnitude')
plt.title('Magnitude vs Incident Angle (C130)')
plt.legend()
plt.grid(True)
plt.show()
