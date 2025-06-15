import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import re
import joblib
import os


# 1. Load dataset
df = pd.read_csv('data.csv')  # replace with your file

# 2. Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['status'])

# 3. Clean and preprocess text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

df['cleaned_statement'] = df['statement'].apply(clean_text)

# 4. Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_statement'], df['label_encoded'], test_size=0.3, random_state=42
)

# 5. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

# 6. Train SVM classifier
svm = LinearSVC()
svm.fit(X_train, train_labels)

# 7. Predict and evaluate
val_preds = svm.predict(X_val)
accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation Accuracy: {accuracy:.4f}")

# 8. Save predictions
df_preds = pd.DataFrame({
    'true_label': val_labels,
    'predicted_label': val_preds
})
df_preds.to_csv('preds/val_predictions_svm.csv', index=False)
print(df_preds.head())

# 9. Save model
os.makedirs('results/SVM', exist_ok=True)
joblib.dump(svm, 'results/SVM/svm_model.joblib')
joblib.dump(vectorizer, 'results/SVM/tfidf_vectorizer.joblib')

# *** To load the model later:
# svm = joblib.load('results/SVM/svm_model.joblib')
# vectorizer = joblib.load('results/SVM/tfidf_vectorizer.joblib')
