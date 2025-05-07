import pandas as pd
df=pd.read_csv("Customer-Churn.csv")
print(df.head())
print(df.info())

#Find Missing Values in Dataset and remove them
print(df.isnull().sum())
df = df.dropna()

#Remaining Missing Values After Cleaning
print(df.isnull().sum())


#data preprocessing


from sklearn.preprocessing import LabelEncoder
categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])
print("\nDataset after Encoding:\n")
print(df.head())
from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = df.drop(columns=['Churn'])  # Independent variables
y = df['Churn']  # Target variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Verify split size
print(f"Training Set Size: {X_train.shape}")
print(f"Testing Set Size: {X_test.shape}")
# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Apply scaling to training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Verify scaling
print("\nSample of Scaled Features:\n", X_train[:5])



# Import Random Forest Classifier



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print(f"\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Define class labels
labels = ["Non-Churn (0)", "Churn (1)"]

# Plot confusion matrix with labeled axes


plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="coolwarm", linewidths=1, square=True, cbar=False)
plt.title("Confusion Matrix - Random Forest", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks([0.5, 1.5], labels, fontsize=10)  # Label Predicted Classes
plt.yticks([0.5, 1.5], labels, fontsize=10)  # Label True Classes
plt.show()

from sklearn.metrics import roc_curve, auc

# Get probability scores
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Compute ROC Curve


fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.legend(loc="lower right")
plt.show()


#compute  precision_recall_curve


from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_prob_rf)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="purple", lw=2, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Random Forest)")
plt.legend()
plt.show()



# Import Logistic Regression



from sklearn.linear_model import LogisticRegression
# Initialize and train the model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# Make predictions
y_pred_log = log_model.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score,classification_report

print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# Compute confusion matrix


conf_matrix_log = confusion_matrix(y_test, y_pred_log)
# Define labels
labels = ["Non-Churn (0)", "Churn (1)"]
# Plot the confusion matrix with labeled axes
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_log, annot=True, fmt="d", cmap="coolwarm", linewidths=1, square=True, cbar=False)
plt.title("Confusion Matrix - Logistic Regression", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks([0.5, 1.5], labels, fontsize=10)  # Label Predicted Classes
plt.yticks([0.5, 1.5], labels, fontsize=10)  # Label True Classes
plt.show()

# Compute ROC curve and AUC


from sklearn.metrics import roc_curve, auc
# Get predicted probabilities
y_prob_log = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_log)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


# Compute precision_recall_curve


from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_prob_log)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="purple", lw=2, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()
