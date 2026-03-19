import os
import pandas as pd
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# --- SETUP ---
NEWS_API_KEY = "b13947d321a24e089c0af84aedfc47a9"  # <--- Paste your key here
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Initialize FinBERT
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentiment(text):
    if not text: return 0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1).numpy()[0]
    return probs[0] - probs[1] # Positive - Negative

# --- FETCH DATA ---
# Set your desired dates (YYYY-MM-DD)
# Note: Free NewsAPI is restricted to the last 30 days.
articles = newsapi.get_everything(
    q='Amazon AND stock',
    from_param='2026-03-01',
    to='2026-03-18',
    language='en',
    sort_by='publishedAt',
    page_size=100
)

data_list = []
for art in articles['articles']:
    # Clean the date to YYYY-MM-DD
    date_str = art['publishedAt'][:10]
    score = get_sentiment(art['title'])
    data_list.append({'Date': date_str, 'Sentiment_Score': score})

# --- AGGREGATE ---
df = pd.DataFrame(data_list)
if not df.empty:
    final_df = df.groupby('Date').agg(
        Sentiment_Score=('Sentiment_Score', 'mean'),
        Article_Count=('Sentiment_Score', 'count')
    ).reset_index()

    final_df.to_csv('amazon_historical_sentiment.csv', index=False)
    print("CSV Created! Summary:")
    print(final_df)
else:
    print("No articles found for those dates.")