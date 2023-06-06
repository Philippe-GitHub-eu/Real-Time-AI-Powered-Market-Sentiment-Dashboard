import os
import dash
import datetime
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import requests
import nltk
import torch
import plotly.graph_objs as go
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import string
from textblob import TextBlob
from dotenv import load_dotenv
import praw
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('omw-1.4')

# Loading environment variables
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
print(API_KEY)  

# Load sentiment analysis model and tokenizer
model_name_sentiment = 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_model = pipeline("sentiment-analysis", model=model_name_sentiment)


# Load GPT-2 model and tokenizer
model_name = 'gpt2'  # Replace with your desired GPT-2 model name
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Check the number of GPUs available
if torch.cuda.is_available():  
    print("Number of GPUs Available: ", torch.cuda.device_count())

    # Load sentiment model to the first GPU
    device1 = torch.device('cuda:0') 
    sentiment_model = sentiment_model.to(device1)

    # Load GPT2 model to the second GPU
    device2 = torch.device('cuda:1')
    model = model.to(device2)



# Set pad_token_id
pad_token_id = tokenizer.eos_token_id  
model.config.pad_token_id = pad_token_id



# Load pretrained language model for text generation
text_generator = pipeline("text-generation", model="gpt2")

# Load NLP components
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


def concatenate_articles(articles):
    # Concatenate the content of all articles into a single string
    article_texts = [article['content'] for article in articles]
    concatenated_text = ' '.join(article_texts)
    return concatenated_text


def get_news_articles(q, from_param, to, language='en', page_size=10):
    url = "https://newsapi.org/v2/everything"

    payload = {
        'q': q,
        'from': from_param,
        'to': to,
        'language': language,
        'apiKey': API_KEY,
    }

    response = requests.get(url, params=payload)
    if response.status_code == 200:
        print(f"Couldn't retrieve articles. HTTP Status Code: {response.status_code}")
        print(f"Response: {response.content}")  # This will print the entire response
        return response.json().get('articles', [])
    else:
        print(f"Couldn't retrieve articles. HTTP Status Code: {response.status_code}")
        return [{'content': 'Error retrieving articles', 'title': 'Error'}]  # Return a default article when there's an error


def get_reddit_posts(subreddit_name, limit=10):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent='script by u/usualwarrior'
    )

    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        posts.append({
            'title': post.title,
            'text': post.selftext
        })
    return posts


# Update the get_sentiment function to use the Transformers model
def get_sentiment(text):
    result = sentiment_model(text)
    sentiment = result[0]['label']
    return sentiment


def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return words


def process_generated_text(generated_text):
    # Extract the generated text from the list
    text = generated_text[0]['generated_text']

    # Perform additional processing on the generated text if needed
    if isinstance(text, str):
        processed_text = text.replace("@", "")  # Remove '@' symbol from the generated text
    else:
        processed_text = ""

    return processed_text


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='query-input', type='text', value='apple'),
    dcc.DatePickerSingle(
        id='date-from-input',
        date=datetime.datetime.now().date() - datetime.timedelta(days=29)  # Update here
    ),
    dcc.DatePickerSingle(
        id='date-to-input',
        date=datetime.datetime.now().date(),
        max_date_allowed=datetime.datetime.now().date()
    ),
    dcc.Input(id='page-size-input', type='number', value=10),
    html.Label('Language'),
    dcc.Dropdown(
        id='language-input',
        options=[
            {'label': 'None', 'value': ''},
            {'label': 'English', 'value': 'en'},
            {'label': 'French', 'value': 'fr'},
            {'label': 'German', 'value': 'de'},
            {'label': 'Spanish', 'value': 'es'},
            {'label': 'Italian', 'value': 'it'},
            {'label': 'Portuguese', 'value': 'pt'},
            {'label': 'Russian', 'value': 'ru'},
            {'label': 'Japanese', 'value': 'ja'},
            {'label': 'Chinese', 'value': 'zh'},
            {'label': 'Dutch', 'value': 'nl'},
            {'label': 'Swedish', 'value': 'sv'},
            {'label': 'Norwegian', 'value': 'no'},
            {'label': 'Danish', 'value': 'da'},
            # Add more language options here
        ],
        value='',  # Set default language
    ),
    dcc.Dropdown(
    id='source-input',
    options=[
        {'label': 'News', 'value': 'news'},
        {'label': 'Reddit', 'value': 'reddit'},
    ],
    value='news'  # Set default source
    ),
    html.Div([
        html.Button('Submit', id='submit-button'),
        dcc.Textarea(id='generated-text-output', style={'width': '100%', 'height': 200}),
    ], style={'display': 'block'}),
    html.Div(id='processed-text-output'),
    dcc.Graph(id='sentiment-histogram'),
    dcc.Graph(id='sentiment-histogram-textblob')
])


@app.callback(
    [Output('sentiment-histogram', 'figure'),
     Output('sentiment-histogram-textblob', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [State('query-input', 'value'),
     State('date-from-input', 'date'),
     State('date-to-input', 'date'),
     State('page-size-input', 'value'),
     State('language-input', 'value'),
     State('source-input', 'value')]
)

def update_output(n_clicks, query, date_from, date_to, page_size, language, source):
    if n_clicks is None:
        return dash.no_update, dash.no_update

    # Check if date_from and date_to are not None before passing them to get_news_articles
    date_from = date_from if date_from is not None else '2023-01-01'  # Provide a default date if date_from is None
    date_to = date_to if date_to is not None else str(datetime.datetime.now().date())  # Provide a default date if date_to is None

    # Fetch data according to the selected source
    if source == 'news':
        articles = get_news_articles(query, date_from, date_to, language=language, page_size=int(page_size))
    elif source == 'reddit':
        articles = get_reddit_posts(query, limit=int(page_size))  # 'query' here is used as subreddit name
        articles = [{'content': post['text'], 'title': post['title']} for post in articles]

    articles = get_news_articles(query, date_from, date_to, language=language, page_size=int(page_size))

    # Check if the articles list is not empty before trying to access its elements
    if not articles:
        return (
            go.Figure(data=[go.Bar(x=['Error'], y=[0])], layout=go.Layout(title='No articles found')),
            go.Figure(data=[go.Bar(x=['Error'], y=[0])], layout=go.Layout(title='No articles found'))
        )

    if articles[0]['title'] == 'Error':
        return (
            go.Figure(data=[go.Bar(x=['Error'], y=[0])], layout=go.Layout(title='Error retrieving articles')),
            go.Figure(data=[go.Bar(x=['Error'], y=[0])], layout=go.Layout(title='Error retrieving articles'))
        )
    else:
        # Concatenate the content of the articles
        concatenated_text = concatenate_articles(articles)

        # Modify the input_text to include the articles' information
        input_text = "Make a Small Market Analysis based on this information: " + concatenated_text

        # Generate text using the language model
        generated_texts = []
        chunk_size = 1024
        for i in range(0, len(input_text), chunk_size):
            chunk = input_text[i:i + chunk_size]
            generated_text = text_generator(chunk, max_length=1024, num_return_sequences=1)
            generated_texts.append(generated_text)

        # Process the generated text if needed
        processed_text = process_generated_text(generated_text)

        # Continue with the rest of the code for sentiment analysis and visualization
        preprocessed_articles = [preprocess_text(article['content']) for article in articles]
        article_sentiments = [get_sentiment(' '.join(article)) for article in preprocessed_articles]

        data = [go.Histogram(x=article_sentiments)]
        figure = {
            'data': data,
            'layout': go.Layout(title='Sentiment Histogram')
        }

        # Perform sentiment analysis using TextBlob
        textblob_sentiments = [TextBlob(' '.join(article)).sentiment.polarity for article in preprocessed_articles]
        data_textblob = [go.Histogram(x=textblob_sentiments)]
        figure_textblob = {
            'data': data_textblob,
            'layout': go.Layout(title='Sentiment Histogram (TextBlob)')
        }

        return figure, figure_textblob


@app.callback(
    Output('processed-text-output', 'children'),  # Update the output component to display the processed text
    [Input('submit-button', 'n_clicks')],
    [State('query-input', 'value'),
     State('date-from-input', 'date'),
     State('date-to-input', 'date'),
     State('page-size-input', 'value'),
     State('language-input', 'value'),
     State('source-input', 'value')]
)
def update_processed_text(n_clicks, query, date_from, date_to, page_size, language, source):
    if n_clicks is None:
        return dash.no_update

    # Check if date_from and date_to are not None before passing them to get_news_articles
    date_from = date_from if date_from is not None else '2023-01-01'  # Provide a default date if date_from is None
    date_to = date_to if date_to is not None else str(datetime.datetime.now().date())  # Provide a default date if date_to is None

    # Fetch data according to the selected source
    if source == 'news':
        articles = get_news_articles(query, date_from, date_to, language=language, page_size=int(page_size))
    elif source == 'reddit':
        articles = get_reddit_posts(query, limit=int(page_size))  # 'query' here is used as subreddit name
        articles = [{'content': post['text'], 'title': post['title']} for post in articles]

    articles = get_news_articles(query, date_from, date_to, language=language, page_size=int(page_size))

    # Check if the articles list is not empty before trying to access its elements
    if not articles:
        return "No articles found."

    if articles[0]['title'] == 'Error':
        return "Error retrieving articles."
    else:
        # Concatenate the content of the articles
        concatenated_text = concatenate_articles(articles)

        # Modify the input_text to include the articles' information
        input_text = "Make an Analysis based on these informations: " + concatenated_text

        # Generate text using the language model
        generated_texts = []
        chunk_size = 1024
        for i in range(0, len(input_text), chunk_size):
            chunk = input_text[i:i + chunk_size]
            generated_text = text_generator(chunk, max_length=1024, num_return_sequences=1)
            generated_texts.append(generated_text)

        # Process the generated text if needed
        processed_text = process_generated_text(generated_text)

        return processed_text if processed_text else "No processed text available."


if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
