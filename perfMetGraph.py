import pandas as pd

import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('/Users/krish/Development/SDC/perf_data/data_with_predicted_angles.csv')

# Extract the actual and predicted values
# forgive header
data = data.iloc[1:, :]
img_address = data.iloc[:, 0]
actual_values = data.iloc[:, 1]
predicted_values = data.iloc[:, 2]

# mae, mse
mae = abs(actual_values - predicted_values)
mae = mae.mean()
print('MAE:', mae)

mse = (actual_values - predicted_values) ** 2
mse = mse.mean()
print('MSE:', mse)

# rmse = (actual_values - predicted_values) ** 2
rmse = mse.mean() ** 0.5
print('RMSE:', rmse)

# calculate R-squared
mean_actual = actual_values.mean()
ss_total = sum((actual_values - mean_actual) ** 2)
ss_res = sum((actual_values - predicted_values) ** 2)
r_squared = 1 - (ss_res / ss_total)
print('R-Squared:', r_squared)


plt.plot(actual_values, label='Actual Angle')
plt.plot(predicted_values, label='Predicted Angle')
plt.xlabel('Images')
plt.ylabel('Angle Value')
plt.title('Actual vs Predicted')

plt.legend()
plt.show()