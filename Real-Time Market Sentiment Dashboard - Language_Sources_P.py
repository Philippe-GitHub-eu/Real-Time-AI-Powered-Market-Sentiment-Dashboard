import os
import dash
import datetime
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob
from dotenv import load_dotenv
import praw

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


def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


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
    value='news',  # Set default source
    ),
    html.Button('Submit', id='submit-button'),
    dcc.Graph(id='sentiment-histogram')

])


@app.callback(
    Output('sentiment-histogram', 'figure'),
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
        return dash.no_update

    date_from = date_from if date_from is not None else '2023-01-01'  
    date_to = date_to if date_to is not None else str(datetime.datetime.now().date())  
    page_size = int(page_size)

    # Add a condition for the source
    if source == 'news':
        articles = get_news_articles(query, date_from, date_to, language=language, page_size=page_size)
        content_key = 'content'
    else:  # source == 'reddit'
        articles = get_reddit_posts(query, limit=page_size)
        content_key = 'text'

    if not articles:
        return go.Figure(data=[go.Bar(x=['Error'], y=[0])],
                         layout=go.Layout(title='No articles found'))

    if articles[0]['title'] == 'Error':
        return go.Figure(data=[go.Bar(x=['Error'], y=[0])],
                         layout=go.Layout(title='Error retrieving articles'))

    preprocessed_articles = [preprocess_text(article[content_key]) for article in articles]
    article_sentiments = [get_sentiment(' '.join(article)) for article in preprocessed_articles]

    data = [go.Histogram(x=article_sentiments)]
    figure = {
        'data': data,
        'layout': go.Layout(title='Sentiment Histogram')
    }

    return figure



if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
