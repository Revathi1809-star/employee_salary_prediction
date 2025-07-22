import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("adult 3.csv")

# Drop missing values
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ✅ Use only 4 features matching app.py
X = df[["age", "hours-per-week", "workclass", "native-country"]]
y = df["income"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "salary_model.pkl")
print("✅ Model saved as salary_model.pkl")
