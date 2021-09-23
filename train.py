import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/data_processed.csv")

### Get features ready to model
X = df.drop(["rating"], axis=1)
Y = df['rating']

### Split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 27)

### Scale the data
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_scaler = StandardScaler()
Y_train = y_scaler.fit_transform(np.array(Y_train).reshape(-1, 1)).squeeze()
Y_test = y_scaler.transform(np.array(Y_test).reshape(-1, 1)).squeeze()

# Train the simplest Linear model
lrm = LinearRegression().fit(X_train, Y_train)

# Capture the metrics
r_2 = lrm.score(X_test, Y_test)
Y_pred = lrm.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

# Record the metrics to the metrics.json file
with open("metrics.json", 'w') as outfile:
        json.dump({ "r_squared": r_2, "rmse": rmse}, outfile)

# Save the model pickle
models_dir = Path('model')
models_dir.mkdir(exist_ok=True)
model_path = models_dir / "model.pkl"
with open(model_path, 'wb') as fp:
    pickle.dump(lrm, fp)
