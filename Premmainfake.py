# Import necessary libraries
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer, TFBertForSequenceClassification

# Download NLTK resources only once
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Streamlit UI setup
st.title('GUVI PROJECT - Fake News Classifier')
st.write("*********Project done by Premila.M ******************************************************")
st.balloons()

# Display an image
file_path = r"C:\Users\premi\Desktop\Premila\projects\Fake\guvi1.png"
image = Image.open(file_path)
st.image(image, caption='', width=300)

# File path for the dataset
data_path = r"C:\Users\premi\Desktop\Premila\projects\Fake\WELFake_Dataset.csv"

# Function to load and preprocess the data
@st.cache_data  # Cache the function to optimize performance
def load_and_preprocess_data(data_path):
    """Load data from CSV file and preprocess it for model training."""
    try:
        # Read data from CSV file
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: The file at {data_path} was not found.")
        return pd.DataFrame()  # Return empty DataFrame if file is not found
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return pd.DataFrame()  # Return empty DataFrame for any other exception

    # Combine title and text into a single content column
    data['content'] = data['title'].fillna('') + " " + data['text'].fillna('')
    data['content'] = data['content'].apply(preprocess_text)  # Apply text preprocessing
    return data

def preprocess_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)  # Lemmatize and remove stopwords
    return text

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    """Plot a confusion matrix for the given model predictions."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)  # Create heatmap
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix for {title}')
    st.pyplot(fig)

# Initialize a dictionary to store performance metrics
performance_metrics = {'Model': [], 'Accuracy': [], 'F1 Score': []}

# Function to update performance metrics
def update_performance_metrics(model_name, y_true, y_pred):
    """Update the performance metrics for a given model."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    performance_metrics['Model'].append(model_name)
    performance_metrics['Accuracy'].append(accuracy)
    performance_metrics['F1 Score'].append(f1)

# Function to train and evaluate Logistic Regression model
def log_reg_model():
    """Train and evaluate a Logistic Regression model."""
    data = load_and_preprocess_data(data_path)
    X = data['content']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()
    
    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Display model evaluation results
    st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    st.write(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, 'Logistic Regression')
    update_performance_metrics('Logistic Regression', y_test, y_pred)

# Function to train and evaluate Support Vector Machine model
def svm_model():
    """Train and evaluate a Support Vector Machine (SVM) model."""
    data = load_and_preprocess_data(data_path)
    X = data['content']
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    # Train SVM model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display model evaluation results
    st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    st.write(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred, 'Support Vector Machine')
    update_performance_metrics('Support Vector Machine', y_test, y_pred)

# Function to train and evaluate Neural Network (LSTM) model
def neural_nt_model():
    """Train and evaluate a Neural Network model with LSTM layers."""
    data = load_and_preprocess_data(data_path)
    X = data['content']
    y = data['label']

    # Tokenize and pad text data
    max_features = 25000
    maxlen = 300
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=128),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Predict and evaluate the model
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    st.write('Neural Network - LSTM Model Evaluation:')
    st.write(classification_report(y_val, y_pred))
    plot_confusion_matrix(y_val, y_pred, 'Neural Network - LSTM Model')
    update_performance_metrics('Neural Network - LSTM Model', y_val, y_pred)

# Function to train and evaluate Random Forest model
def random_forest_model():
    """Train and evaluate a Random Forest model."""
    data = load_and_preprocess_data(data_path)
    X = data['content']
    y = data['label']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Display model evaluation results
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, 'Random Forest Model')

    # Update performance metrics
    update_performance_metrics('Random Forest Model', y_test, y_pred)

# Function to visualize and compare performance metrics of different models
def visualize_performance():
    """Visualize the performance metrics of all models."""
    # Convert the performance metrics dictionary to a DataFrame
    df_metrics = pd.DataFrame(performance_metrics)
    
    # Plot performance metrics using bar charts
    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Accuracy', data=df_metrics, ax=ax, color='skyblue', label='Accuracy')
    sns.barplot(x='Model', y='F1 Score', data=df_metrics, ax=ax, color='orange', label='F1 Score')

    # Customize plot appearance
    ax.set_title('Model Performance Comparison')
    ax.legend()
    st.pyplot(fig)

# Main function for Streamlit app
def main():
    """Main function to drive the Streamlit application."""
    # Sidebar for selecting the model to review
    option = st.sidebar.selectbox(
        'Which model do you want to review?', 
        ['Home', 'Logistic Regression', 'Support Vector Machine', 'Neural Network - LSTM Model', 'Random Forest Model', 'Visualize Performance', 'Exit']
    )
    
    # Display content based on user choice
    if option == 'Home':
        st.write("""
        In today’s extremely interconnected world, the spread of fake news has become a major concern.
        This Data Science project aims to combat this spread of misinformation by creating a model that detects fake news.
        """)
        st.subheader('Packages Used in Project:')
        st.write('Python packages like Pandas, NumPy, and scikit-learn have been used as the foundation of this project.') 
        st.write('Here we have utilized a dataset "WELFake_Dataset.csv" to train and evaluate our model.')
    
    elif option == 'Logistic Regression':
        log_reg_model()

    elif option == 'Support Vector Machine':
        svm_model()
    
    elif option == 'Neural Network - LSTM Model':
        neural_nt_model()
    
    elif option == 'Random Forest Model':
        random_forest_model()

    elif option == 'Visualize Performance':
        visualize_performance()
    
    elif option == 'Exit':
        st.write('Goodbye!')
        st.stop()

# Execute the main function when the script is run
if __name__ == '__main__':
    main()
