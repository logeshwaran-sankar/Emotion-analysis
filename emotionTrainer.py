import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================
# PART 1: EMOTION PREDICTION (MULTI-CLASS)
# =====================================================

print("\n========== EMOTION PREDICTION ==========")

# -------- Load emotion datasets (EXACT columns) --------
df_e1 = pd.read_csv("emotion_data_1.csv")[['text', 'label']]
df_e1.rename(columns={'label': 'Emotion'}, inplace=True)

df_e2 = pd.read_csv("emotion_data_2.csv")[['Sentence', 'Label']]
df_e2.rename(columns={'Sentence': 'text', 'Label': 'Emotion'}, inplace=True)

df_e3 = pd.read_csv("emotion_data_3.csv")[['Text', 'Emotion']]
df_e3.rename(columns={'Text': 'text'}, inplace=True)

# Merge
emotion_df = pd.concat([df_e1, df_e2, df_e3], ignore_index=True)

# -------- Clean --------
emotion_df.dropna(inplace=True)
emotion_df.drop_duplicates(subset='text', inplace=True)

emotion_df['text'] = emotion_df['text'].str.lower().str.strip()
emotion_df['Emotion'] = emotion_df['Emotion'].str.lower().str.strip()

valid_emotions = [
    'neutral', 'joy', 'sadness', 'fear',
    'surprise', 'anger', 'shame', 'disgust'
]

emotion_df = emotion_df[emotion_df['Emotion'].isin(valid_emotions)]

print("Emotion dataset shape:", emotion_df.shape)
print("Emotion distribution:\n", emotion_df['Emotion'].value_counts())

# -------- Visualization: Emotion distribution --------
plt.figure(figsize=(8, 5))
sns.countplot(
    data=emotion_df,
    x='Emotion',
    order=emotion_df['Emotion'].value_counts().index
)
plt.title("Emotion Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------- Encode labels --------
emotion_le = LabelEncoder()
emotion_df['label'] = emotion_le.fit_transform(emotion_df['Emotion'])

X_e = emotion_df['text']
y_e = emotion_df['label']

# -------- Train-test split --------
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_e, y_e, test_size=0.2, random_state=42, stratify=y_e
)

# -------- TF-IDF --------
emotion_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=7000
)

X_train_e_vec = emotion_vectorizer.fit_transform(X_train_e)
X_test_e_vec = emotion_vectorizer.transform(X_test_e)

# -------- Train model --------
emotion_model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs'
)
emotion_model.fit(X_train_e_vec, y_train_e)

# -------- Evaluation --------
y_pred_e = emotion_model.predict(X_test_e_vec)

print("\nEmotion Accuracy:", accuracy_score(y_test_e, y_pred_e))
print(classification_report(
    y_test_e, y_pred_e, target_names=emotion_le.classes_
))

cm_e = confusion_matrix(y_test_e, y_pred_e)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_e,
    annot=True,
    fmt='d',
    xticklabels=emotion_le.classes_,
    yticklabels=emotion_le.classes_,
    cmap='Blues'
)
plt.title("Emotion Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------- Save emotion model --------
joblib.dump(emotion_model, "emotion_model.pkl")
joblib.dump(emotion_vectorizer, "emotion_vectorizer.pkl")
joblib.dump(emotion_le, "emotion_label_encoder.pkl")

print("✅ Emotion model saved")

# =====================================================
# PART 2: DEPRESSION DETECTION (BINARY)
# =====================================================

print("\n========== DEPRESSION DETECTION ==========")

# -------- Load cleaned depression dataset --------
dep_df = pd.read_csv("final_depression_dataset.csv")

print("Depression dataset shape:", dep_df.shape)
print("Label distribution:\n", dep_df['label'].value_counts())

X_d = dep_df['text'].str.lower().str.strip()
y_d = dep_df['label']

# -------- Train-test split --------
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
)

# -------- TF-IDF --------
dep_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_d_vec = dep_vectorizer.fit_transform(X_train_d)
X_test_d_vec = dep_vectorizer.transform(X_test_d)

# -------- Train model --------
dep_model = LogisticRegression(max_iter=1000)
dep_model.fit(X_train_d_vec, y_train_d)

# -------- Evaluation --------
y_pred_d = dep_model.predict(X_test_d_vec)

print("\nDepression Accuracy:", accuracy_score(y_test_d, y_pred_d))
print(classification_report(y_test_d, y_pred_d))

cm_d = confusion_matrix(y_test_d, y_pred_d)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_d,
    annot=True,
    fmt='d',
    cmap='Reds'
)
plt.title("Depression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------- Save depression model --------
joblib.dump(dep_model, "depression_model.pkl")
joblib.dump(dep_vectorizer, "depression_vectorizer.pkl")

print("✅ Depression model saved")
print("\n🎉 BOTH MODELS TRAINED SUCCESSFULLY")
