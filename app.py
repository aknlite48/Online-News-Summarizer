import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from flask import Flask, request, jsonify, render_template

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

# Load the fine-tuned model and tokenizer
model_path = "./t5-summarizer"  # Path to the saved model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Define a summarization function
def summarize_text(text, max_input_length=512, max_output_length=200, min_output_length=40):
    # Tokenize the input text
    inputs = tokenizer(
        f"summarize: {text}", 
        return_tensors="pt", 
        max_length=max_input_length, 
        truncation=True
    )
    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_output_length, 
        min_length=min_output_length, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def is_url(text):
    """
    Check if the given text appears to be a URL.
    
    Args:
        text (str): The text to check
        
    Returns:
        bool: True if the text appears to be a URL, False otherwise
    """
    # Handle empty or non-string input
    if not text or not isinstance(text, str):
        return False
    
    # Common URL patterns
    url_pattern = re.compile(
        r'^(?:http[s]?://)?'  # http:// or https:// (optional)
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ip address
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    # Quick pattern match
    if not url_pattern.match(text):
        return False
    
    try:
        # Additional validation using urlparse
        result = urlparse(text)
        # Check if at least scheme or netloc (domain) exists
        return bool(result.scheme or result.netloc)
    except:
        return False

def scrape_article(url):
    """
    Scrapes article text from a news website URL.
    
    Args:
        url (str): The URL of the news article
        
    Returns:
        dict: Contains title, author, date, and content of the article
    """
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Fetch the webpage
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Initialize dictionary for article data
        article_data = {
            'title': '',
            'author': '',
            'date': '',
            'content': ''
        }
        
        # Extract title (common patterns)
        title_tags = soup.find(['h1', 'h2'], class_=re.compile('(title|headline)', re.I))
        if title_tags:
            article_data['title'] = title_tags.get_text().strip()
            
        # Extract author (common patterns)
        author_tags = soup.find(['a', 'span', 'div'], class_=re.compile('(author|byline)', re.I))
        if author_tags:
            article_data['author'] = author_tags.get_text().strip()
            
        # Extract date (common patterns)
        date_tags = soup.find(['time', 'span', 'div'], class_=re.compile('(date|published|time)', re.I))
        if date_tags:
            article_data['date'] = date_tags.get_text().strip()
        
        # Extract main content
        # First try with article tag
        article_content = soup.find('article')
        
        if not article_content:
            # Try common content div patterns
            article_content = soup.find(['div', 'section'], class_=re.compile('(article|content|story)', re.I))
        
        if article_content:
            # Remove unwanted elements
            unwanted = article_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside'])
            for elem in unwanted:
                elem.decompose()
                
            # Extract paragraphs
            paragraphs = article_content.find_all('p')
            content = '\n\n'.join(p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50)
            article_data['content'] = content
            
        return article_data
        
    except Exception as e:
        return f"Error scraping article: {str(e)}"

def clean_text(text):
    """
    Cleans extracted text by removing extra whitespace and unwanted characters.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

app = Flask(__name__)
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# @app.route("/summarize", methods=["POST"])
# def summarize():
#     data = request.json
#     text = data.get("text", "")
#     if not text:
#         return jsonify({"error": "No text provided."}), 400

#     summary = summarize_text(model, tokenizer, text)
#     return jsonify({"summary": summary})

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        # Get input data from the request
        data = request.get_json()
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "Input text is missing!"}), 400

        if is_url(text):
            article = scrape_article(text)
            if isinstance(article, dict):
                text = article['content']
                if (text==''):
                    return jsonify({"error": 'Unable to read link'}), 500
                summary = summarize_text("summarize: "+text)
                return jsonify({"summary": summary})
        else:

            # Perform summarization
            summary = summarize_text("summarize: "+text)
            return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)