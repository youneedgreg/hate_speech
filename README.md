# Kenya Hate Speech Classification Model

This project builds a machine learning model to classify text as "Hate Speech," "Offensive," or "Neither" using the Kenya Hate Speech dataset.

## Dataset Overview

The dataset contains 48,076 tweets with the following structure:
- `hate_speech`: Integer flag (1 if hate speech)
- `offensive_language`: Integer flag (1 if offensive)
- `neither`: Integer flag (1 if neither)
- `Class`: Integer class label (0 = Neither, 1 = Offensive, 2 = Hate Speech)
- `Tweet`: String containing the tweet text

## Setup and Requirements

1. **Install required packages:**
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn nltk
   ```

2. **Download the dataset:**
   - Get the dataset from Kaggle: https://www.kaggle.com/datasets/edwardombui/hatespeech-kenya
   - Save it as `HateSpeech_Kenya.csv` in your project directory

## Running the Model

The main script `hate_speech_classification_model.py` performs the following steps:

1. **Data loading and exploration**
2. **Text preprocessing**
3. **Feature engineering**
4. **Model training and evaluation**
5. **Hyperparameter tuning**
6. **Feature importance analysis**
7. **Error analysis**
8. **Model saving**
9. **Classification function creation**

To run the complete pipeline:
```
python hate_speech_classification_model.py
```

## Using the Trained Model

After running the script, you'll have:
- `hate_speech_model.pkl`: The trained model
- `tfidf_vectorizer.pkl`: The vectorizer for converting text to features
- `model_info.pkl`: Model metadata and class mapping

Use the provided classification function to classify new text:

```python
def classify_hate_speech(text):
    """
    Classify a given text as Hate Speech, Offensive, or Neither
    
    Args:
        text (str): The text to classify
        
    Returns:
        str: The predicted class label
        float: Confidence score (if available)
    """
    # Load the model and vectorizer
    with open('hate_speech_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open('model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform the text
    text_tfidf = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get confidence score if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(text_tfidf)[0]
        confidence = max(probas)
    
    # Convert to label
    predicted_label = model_info['class_mapping'][prediction]
    
    if confidence is not None:
        return predicted_label, confidence
    else:
        return predicted_label, None

# Example usage
text = "I love how diverse our community is!"
prediction, confidence = classify_hate_speech(text)
print(f"Text: '{text}'")
print(f"Prediction: '{prediction}'") 
if confidence:
    print(f"Confidence: {confidence:.2f}")
```

## Outputs

The script generates several visualization files:
- `class_distribution.png`: Distribution of classes in the dataset
- `confusion_matrix_*.png`: Confusion matrices for each model
- `model_comparison.png`: Accuracy comparison of different models
- `feature_importance_*.png`: Important features for each class
- `misclassified_examples.csv`: Examples the model got wrong

## Key Features

1. **Customized preprocessing for this dataset:**
   - Handles the specific formatting of tweets
   - Removes usernames (USERNAME_X)
   - Handles special characters and list formatting

2. **Model comparison:**
   - Logistic Regression
   - Random Forest
   - Linear SVM
   - Multinomial Naive Bayes

3. **Comprehensive evaluation:**
   - Accuracy, precision, recall, F1-score
   - Confusion matrices
   - Feature importance analysis
   - Error analysis

## Performance Notes

- The model typically achieves 80-85% accuracy on the test set
- "Neither" class is usually better classified than "Hate Speech"
- Important features often include ethnicity terms specific to the Kenyan context

## Further Improvements

Consider these enhancements for better performance:
- Try more advanced text representation (Word2Vec, BERT embeddings)
- Implement data augmentation for the minority classes
- Add contextual features (e.g., user information, tweet context)
- Implement an ensemble model combining multiple classifiers