# ###############VERSION 01
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.optimizers import Adam
# import numpy as np
# import time
# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# start_time = time.time()

# # Load data (same as before)
# data = pd.read_csv('/Users/Admin/Desktop/dataset/BGR_training/compdata.csv')
# X = data.iloc[:, :-1].values
# y = data.iloc[:, -1].values

# # Data splitting (same as before)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature scaling (same as before)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Hyperparameter optimization with k-fold cross validation (same as before)
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# mse_scores = []
# r2_scores = []
# best_mse = float('inf')

# for train_idx, val_idx in kfold.split(X_train):
#     X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
#     y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

#     # Optimized ANN model architecture
#     model = Sequential([
#         Dense(128, activation='relu', input_shape=(X_train.shape[1],)), 
#         Dropout(0.2),   # Add dropout for regularization
#         Dense(64, activation='relu'),
#         Dropout(0.2),  
#         Dense(32, activation='relu'),
#         Dense(1)       
#     ])

#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

#     model.fit(X_train_fold, y_train_fold, epochs=200, batch_size=32, validation_data=(X_val_fold, y_val_fold), 
#               callbacks=[early_stopping], verbose=0)

#     y_pred_fold = model.predict(X_val_fold).flatten()
#     mse_fold = mean_squared_error(y_val_fold, y_pred_fold)
#     r2_fold = r2_score(y_val_fold, y_pred_fold)

#     mse_scores.append(mse_fold)
#     r2_scores.append(r2_fold)

#     if mse_fold < best_mse:
#         best_mse = mse_fold
#         best_model = model  # Save the best model

# print("Average MSE:", np.mean(mse_scores))
# print("Average R-squared:", np.mean(r2_scores))

# # Evaluate the best model on the test set (use best_model)
# y_pred = best_model.predict(X_test).flatten()  # Use best_model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Test MSE:", mse)
# print("Test R-squared:", r2)


# # Visualize the results (use best_model)
# plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
# plt.scatter(range(len(y_pred)), y_pred, color='r', label='Predicted')
# plt.xlabel('Sample Index')
# plt.ylabel('Magnitude')
# plt.title('Artificial Neural Network Regression')
# plt.legend()
# plt.show()

# # Add predicted values to DataFrame and save (use best_model)
# data['predicted_magnitude'] = best_model.predict(scaler.transform(data.iloc[:, :-1].values)).flatten()  
# data.to_csv('/Users/Admin/Desktop/dataset/BGR_training/ann_compdata_predicted.csv', index=False)

# # Save the best model
# best_model.save('/Users/Admin/Desktop/best_ann_model.h5')  # Save best_model

# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time:", execution_time, "seconds")








#######################VERSION 02
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, schedules

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

start_time = time.time()

# Load data (replace with your actual path)
data = pd.read_csv('/Users/Admin/Desktop/dataset/BGR_training/compdata.csv') 
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter optimization with k-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scores = []
r2_scores = []
best_mse = float('inf')
best_model = None

for train_idx, val_idx in kfold.split(X_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # Use Input layer to define input shape
    model = Sequential([
        Input(shape=(X_train.shape[1],)), 
        Dense(128, activation='relu'),
        Dropout(0.2),  
        Dense(64, activation='relu'),
        Dropout(0.2), 
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Learning Rate Strategies with Schedules
    strategies = {
        "Constant LR": Adam(learning_rate=0.001),
        "Exponential Decay": Adam(learning_rate=schedules.ExponentialDecay(
            initial_learning_rate=0.001, decay_steps=10000, decay_rate=0.9)),
        "ReduceLROnPlateau": Adam(learning_rate=0.001),  # Combined with callback below
    }

    for strategy_name, optimizer in strategies.items():
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
        if strategy_name == "ReduceLROnPlateau":
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001))

        model.fit(X_train_fold, y_train_fold, epochs=200, batch_size=32, validation_data=(X_val_fold, y_val_fold), 
                  callbacks=callbacks, verbose=0)

        y_pred_fold = model.predict(X_val_fold).flatten()
        mse_fold = mean_squared_error(y_val_fold, y_pred_fold)

        if mse_fold < best_mse:
            best_mse = mse_fold
            best_model = model
            best_strategy = strategy_name

print(f"Best Strategy: {best_strategy}")

print("Average MSE:", np.mean(mse_scores))
print("Average R-squared:", np.mean(r2_scores))

# Evaluate the best model on the test set (use best_model)
y_pred = best_model.predict(X_test).flatten()  # Use best_model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Test MSE:", mse)
print("Test R-squared:", r2)


# Visualize the results (use best_model)
plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
plt.scatter(range(len(y_pred)), y_pred, color='r', label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Artificial Neural Network Regression')
plt.legend()
plt.show()
    

# Add predicted values to DataFrame and save (use best_model)
data['predicted_magnitude'] = best_model.predict(scaler.transform(data.iloc[:, :-1].values)).flatten()  
data.to_csv('/Users/Admin/Desktop/dataset/BGR_training/ann_compdata_predicted.csv', index=False)

# Save the best model
best_model.save('/Users/Admin/Desktop/best_annr_model.h5')  # Save best_model




