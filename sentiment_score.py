import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import csv
import nltk
nltk.download('vader_lexicon')

# Scraping data from a given URL
def scrape_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text_data = []

    # Extracting text from various HTML elements 
    for tag in soup.find_all(['p', 'div', 'span']):
        text_data.append(tag.get_text(strip=True))

    return text_data

# Performing sentiment analysis 
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']

    # Classify the sentiment as positive, negative, or neutral
    if sentiment_score >= 0.05:
        return 'positive'
    elif sentiment_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# URLs for different indices of NIFTY 
urls = [
    'https://www.google.com/finance/quote/NIFTY_50:INDEXNSE',
    ]

# Scrape textual data using each URL and perform sentiment analysis for each URL
for url in urls:
    text_data = scrape_text(url)

    # Perform sentiment analysis and store results
    sentiment_results = []

    for text in text_data:
        sentiment = analyze_sentiment(text)
        sentiment_results.append({'text': text, 'sentiment': sentiment})

    # Visualize the sentiment distribution
    sentiments, counts = np.unique([result['sentiment'] for result in sentiment_results], return_counts=True)
    plt.bar(sentiments, counts, color=['green', 'red', 'blue'])
    plt.title(f'Sentiment Distribution for {url}')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    # Display the overall sentiment distribution
    overall_sentiments, overall_counts = np.unique([result['sentiment'] for result in sentiment_results], return_counts=True)
    overall_percentage = (overall_counts / len(sentiment_results)) * 100

    print(f'\nOverall Sentiment Distribution for {url}:')
    for sentiment, count, percentage in zip(overall_sentiments, overall_counts, overall_percentage):
        print(f"{sentiment.capitalize()}: Count={count}, Percentage={percentage:.2f}%")

    # Saving sentiment results 
    sanitized_url = url.split("/")[-1].replace(":", "_").replace(".", "_")
    csv_file_path = f'sentiment_results_{sanitized_url}.csv'

    # Append .csv extension if not present
    if not csv_file_path.endswith('.csv'):
        csv_file_path += '.csv'

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['text', 'sentiment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the sentiment results
        writer.writerows(sentiment_results)

    print(f'\nSentiment results saved to: {csv_file_path}')
