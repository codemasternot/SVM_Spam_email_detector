Adjustable Risk SMS Spam Filter
Description
This project implements an intelligent SMS spam filter with adjustable risk levels using Machine Learning and Natural Language Processing techniques. 
It classifies messages as either spam or ham (not spam) with customizable threshold settings that allow service teams to balance between:

Low Risk - Prioritizes blocking spam 
Medium Risk - Balanced approach for general use
High Risk - Prioritizes legitimate message delivery

The model is trained on the SMS Spam Collection dataset using various classification algorithms and can achieve F1 scores exceeding 0.91 on test data.
Features

Text preprocessing pipeline with stemming and stopword removal
Multiple classification models (Naive Bayes, SVM, Random Forest, Logistic Regression)
Adjustable risk thresholds for different business needs
Detailed performance metrics and visualizations
Simple API for integration with other systems
User-friendly interface for service team operations

Project Structure
This project follows the CRISP-DM methodology and is organized as follows:
spam-filter-project/
├── data/                  # Data files (raw and processed)
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code modules
├── tests/                 # Unit and integration tests
├── models/                # Saved model files
├── reports/               # Generated analysis and visualizations
└── app/                   # Application code for deployment
Dependencies

Python 3.8+
scikit-learn
NLTK
pandas
NumPy
matplotlib
seaborn
joblib

Installation & Usage
Setup
bash# Clone repository
git clone https://github.com/codemasternot/SVM_Spam_email_detector.git
cd sms-spam-filter

Possible deployment options
Web application (Flask/FastAPI)
Containerized service (Docker)
Cloud function (AWS Lambda, Google Cloud Functions)
Local desktop application

Author

Stephen Moorcroft

License
(c) 2025 Stephen Moorcorft
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, and/or sublicense.
