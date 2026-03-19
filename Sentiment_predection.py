'''
INSTRUCTIONS:

For this to work, need to set up training data
so the model is looking at past news to predict future returns.

'''







import tensorflow as tf
from tensorflow.keras import layers, models

def build_sentiment_fnn():
    model = models.Sequential([
        # Input: The FinBERT score (single value between -1 and 1)
        layers.Input(shape=(1,)), 
        
        # Dense Layers: We don't need many neurons for a single input
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        
        # Output: Predicting the 'Return' (the % change)
        layers.Dense(1, activation='linear') 
    ])

    model.compile(
        optimizer='adam', 
        loss='mse',       # Mean Squared Error for regression
        metrics=['mae']   # Mean Absolute Error to see how 'off' we are in %
    )
    return model

# Initialize your model
sentiment_model = build_sentiment_fnn()
sentiment_model.summary()