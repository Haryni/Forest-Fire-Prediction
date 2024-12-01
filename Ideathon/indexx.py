import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'C:\Users\HARYNI\OneDrive\Desktop\Ideathon\dataset.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Step 1: Data Cleaning and Preprocessing

# Remove columns with all missing values
data_cleaned = data.dropna(axis=1, how='all')

# Impute missing values for numeric columns with the mean
numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data_cleaned.loc[:, numeric_cols] = imputer.fit_transform(data_cleaned[numeric_cols])

# Convert non-numeric categorical columns if present
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
label_enc = LabelEncoder()
for col in categorical_cols:
    if col != 'Forest Fire Occurred':  # Skip the target column
        data_cleaned[col] = label_enc.fit_transform(data_cleaned[col].astype(str))

# Convert the target column "Forest Fire Occurred" to binary format (0 or 1)
data_cleaned['Forest Fire Occurred'] = data_cleaned['Forest Fire Occurred'].apply(lambda x: 1 if x == "Yes" else 0)

# Step 2: Remove Date/Datetime Columns if any
data_cleaned = data_cleaned.select_dtypes(exclude=['datetime64'])

# Separate features and target variable
X = data_cleaned.drop('Forest Fire Occurred', axis=1)
y = data_cleaned['Forest Fire Occurred']

# Step 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Build Neural Network Model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test), verbose=1)

# Step 5: Model Prediction (Yes/No)

# Make predictions and classify as "Yes" (1) or "No" (0)
y_pred_prob = model.predict(X_test)
y_pred = ["Yes" if prob > 0.5 else "No" for prob in y_pred_prob.flatten()]

# Convert y_test to "Yes" or "No" for comparison
y_test_labels = ["Yes" if label == 1 else "No" for label in y_test]

# Display results
print("Predictions:", y_pred[:10])  # Display first 10 predictions for review
print("Actual:", y_test_labels[:10])