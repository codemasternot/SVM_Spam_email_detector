#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import numpy as np


# In[2]:


file_path = "T3\\SMSSpamCollection.csv"  

df = pd.read_csv(file_path, sep='\t', names=['label', 'message'], encoding='utf-8')


# In[3]:


print("\n===== DATASET OVERVIEW =====")
print(f"Dataset shape: {df.shape}")
print(f"Number of messages: {len(df)}") # For data understanding


# In[4]:


# Display the first five rows
print("\n===== FIRST 5 ROWS =====")
print(df.head())


# In[5]:


# Check for number of missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# In[6]:


print("\n===== CLASS DISTRIBUTION =====")
class_counts = df['label'].value_counts()
print(class_counts)
print(f"Class distribution (percentage):")
print(100 * class_counts / len(df))


# In[7]:


# Create a bar chart for class distribution
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='label', data=df)
plt.title('Distribution of Spam vs. Ham (Non-Spam) Messages')
plt.xlabel('Message Category')
plt.ylabel('Count')


# In[8]:


for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'bottom')

plt.tight_layout()
plt.savefig('class_distribution.png')
print("Saved class distribution chart as 'class_distribution.png'")

print("\n===== MESSAGE LENGTH ANALYSIS =====")
df['message_length'] = df['message'].apply(len)
print("Message length statistics:")
print(df.groupby('label')['message_length'].describe())


# In[9]:


# Create a boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='message_length', data=df)
plt.title('Message Length Distribution by Class')
plt.xlabel('Message Category')
plt.ylabel('Message Length (characters)')
plt.tight_layout()
plt.savefig('message_length_distribution.png')
print("Saved message length distribution chart as 'message_length_distribution.png'")


# In[10]:


print("\n===== COMMON WORDS ANALYSIS =====")

def clean_text(text):
    """Basic cleaning of text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

spam_texts = df[df['label'] == 'spam']['message'].apply(clean_text)
ham_texts = df[df['label'] == 'ham']['message'].apply(clean_text)


# In[11]:


# Get word frequency
def get_word_freq(texts):
    all_words = ' '.join(texts).split()
    return Counter(all_words)

spam_word_freq = get_word_freq(spam_texts)
ham_word_freq = get_word_freq(ham_texts)

print("Top 15 words in spam messages:")
print(spam_word_freq.most_common(15))

print("\nTop 15 words in ham (non-spam) messages:")
print(ham_word_freq.most_common(15))


# In[12]:


# Message examples
print("\n===== MESSAGE EXAMPLES =====")
print("Sample spam messages:")
print(df[df['label'] == 'spam']['message'].sample(3).to_string())
print("\nSample ham (non-spam) messages:")
print(df[df['label'] == 'ham']['message'].sample(3).to_string())

print("\nExploration complete!")


# In[ ]:




