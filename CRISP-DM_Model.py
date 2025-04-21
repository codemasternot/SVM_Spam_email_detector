#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import joblib
import nltk
nltk.download('stopwords')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# In terms of the CRISP-DM

# 1. BUSINESS UNDERSTANDING
# Our goal is to build a spam filter with adjustable risk levels
# We'll evaluate models on their ability to classify messages as spam or ham
# We'll implement different risk thresholds to meet business requirements

# 2. DATA UNDERSTANDING
print("\n===== DATA UNDERSTANDING =====")

file_path = "T3\\SMSSpamCollection.csv"
df = pd.read_csv(file_path, sep='\t', names=['label', 'message'], encoding='utf-8')

print(f"Dataset shape: {df.shape}")
print(f"Number of messages: {len(df)}")
print(f"Missing values: {df.isnull().sum().sum()}")


# In[4]:


class_counts = df['label'].value_counts()
print("\nClass distribution:")
print(class_counts)
print(f"Spam ratio: {class_counts['spam'] / len(df):.2%}")


# In[5]:


plt.figure(figsize=(8, 5))
ax = sns.countplot(x='label', data=df)
plt.title('Distribution of Spam vs. Ham Messages')
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')
plt.tight_layout()
plt.savefig('class_distribution.png')


# In[6]:


df['message_length'] = df['message'].apply(len)
print("\nMessage length statistics by class:")
print(df.groupby('label')['message_length'].describe())


# In[7]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='message_length', data=df)
plt.title('Message Length Distribution by Class')
plt.tight_layout()
plt.savefig('message_length_distribution.png')


# In[8]:


# 3. DATA PREPARATION
print("\n===== DATA PREPARATION =====")


# In[9]:


def preprocess_text(text):
    """
    Clean and preprocess text data:
    - Convert to lowercase
    - Remove punctuation
    - Remove numbers
    - Remove stopwords
    - Apply stemming
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Join words back into text
    return ' '.join(words)


# In[14]:



# Apply text preprocessing
df['clean_message'] = df['message'].apply(preprocess_text)


# In[15]:


# 4. FEATURE EXTRACTION & MODEL TRAINING
print("\n===== FEATURE EXTRACTION & MODEL TRAINING =====")

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])

# Encode labels
y = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize and train model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict on test set
y_pred = nb_model.predict(X_test)


# In[16]:


# 5. EVALUATION with confusion matrix
print("\n===== MODEL EVALUATION =====")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

plt.show()


# In[17]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Check other models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (Linear Kernel)': SVC(kernel='linear', probability=True)
}

results = []

# Evaluate each model
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1 Score': f1_score(y_test, preds)
    })
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=['ham', 'spam']))

# Include Naive Bayes results
nb_preds = nb_model.predict(X_test)
results.append({
    'Model': 'Multinomial Naive Bayes',
    'Accuracy': accuracy_score(y_test, nb_preds),
    'Precision': precision_score(y_test, nb_preds),
    'Recall': recall_score(y_test, nb_preds),
    'F1 Score': f1_score(y_test, nb_preds)
})

results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)

print("\n===== MODEL COMPARISON =====")
print(results_df)

# Save to computer to see
results_df.to_csv("model_comparison.csv", index=False)


# In[18]:


# Re-train the SVM with probability enabled
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Saves model and vectorizer
joblib.dump(svm_model, "svm_spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# In[40]:


def predict_with_threshold(model, X_data, threshold=0.5):
    """
    Predict spam (1) or ham (0) with a custom threshold on the spam probability.
    """
    probs = model.predict_proba(X_data)[:, 1]  # Probability of class 'spam'
    return (probs >= threshold).astype(int)

# To see different risk levels
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]  

for thresh in thresholds:
    print(f"\n=== Evaluation at Threshold {thresh} ===")
    preds = predict_with_threshold(svm_model, X_test, threshold=thresh)
    print(classification_report(y_test, preds, target_names=['ham', 'spam']))

    conf_matrix = confusion_matrix(y_test, preds)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'Confusion Matrix (Threshold {thresh})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'conf_matrix_threshold_{int(thresh*100)}.png')
    plt.show()


# In[20]:


# Load the saved model and vectorizer
svm_model = joblib.load("svm_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# Prediction made here
def test_message(msg, threshold=0.5):
    cleaned = preprocess_text(msg)
    vect_msg = vectorizer.transform([cleaned])
    prob = svm_model.predict_proba(vect_msg)[0][1]
    prediction = 'spam' if prob >= threshold else 'ham'
    print(f"\nMessage: {msg}")
    print(f"Predicted: {prediction.upper()} (probability: {prob:.2%} at threshold {threshold})")
    return prediction, prob


# In[39]:


# For testing
test_message("You won a free iPhone! Claim your prize now", threshold=0.4)
test_message("Oh I hear you were talking smack, next time i see you it's on sight.", threshold=0.4)
test_message("Get cheap loans approved instantly, almost interest free.", threshold=0.4)
test_message("Let's meet for lunch tomorrow.", threshold=0.4)
test_message("You've won a $500 gift card! Click on the link to claim. Send bitcoin to recieve.", threshold=0.4)
test_message("We training chest and tri's tommorow bruh? Time to get swole.", threshold=0.4)


# In[ ]:


#6 Deploy in future possibly with Github actions or some other alternative like an API

