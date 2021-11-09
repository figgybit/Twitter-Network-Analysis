import jsonpickle
import tweepy
import networkx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
import datetime
import csv
import copy
import pandas as pd
import re
from wordcloud import WordCloud

import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

import gensim.corpora as corpora
from pprint import pprint
# import pyLDAvis.gensim
import pyLDAvis.gensim_models as gensimvis
import pickle 
import pyLDAvis
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]


def preprocess(all_tweets):
    # Remove punctuation
    # all_tweets[all_tweets.tweet_processed.str.contains('tco',case=False)]

    all_tweets['tweet_processed'] = all_tweets['tweet'].map(lambda x: x.replace("\n", ""))
    all_tweets['tweet_processed'] = all_tweets['tweet_processed'].map(lambda x: x.replace('https://t.co', ''))
    all_tweets['tweet_processed'] = all_tweets['tweet_processed'].map(lambda x: x.replace('https://', ''))
    all_tweets['tweet_processed'] = all_tweets['tweet_processed'].map(lambda x: x.replace('http://', ''))
    all_tweets['tweet_processed'] = all_tweets['tweet_processed'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    all_tweets['tweet_processed'] = all_tweets['tweet_processed'].map(lambda x: x.lower())
    # Print out the first rows of papers
    print(all_tweets['tweet_processed'].head())
    return all_tweets

def build_word_cloud(all_tweets):
    # Join the different processed titles together.
    long_string = ','.join(list(all_tweets['tweet_processed'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image().show()


def lda(all_tweets, num_topics):
    data = all_tweets.tweet_processed.values.tolist()
    data_words = list(sent_to_words(data))# remove stop words

    data_words = remove_stopwords(data_words)
    print(data_words[:1][0][:30])
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    print(corpus[:1][0][:30])
    lda_model = gensim.models.LdaMulticore(corpus=corpus,id2word=id2word,num_topics=num_topics)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    # pyLDAvis.enable_notebook()
    LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(num_topics))
    if 1 == 1:
        LDAvis_prepared = gensimvis.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)# load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(num_topics) +'.html')
    return LDAvis_prepared


def collect_data(mention, maxTweets, consumer_key, consumer_secret):
    auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
    auth.secure = True
    api = tweepy.API(auth)
    searchQuery = f"{mention}"
    retweet_filter='-filter:retweets'
    searchQuery=searchQuery+retweet_filter
    tweetsPerQry = 100
    sinceId = None
    max_id = -1
    tweetCount = 0
    all_tweets = []
    print("Downloading max {0} tweets".format(maxTweets))
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search_tweets(q=searchQuery, count=tweetsPerQry, tweet_mode = 'extended')
                else:
                    new_tweets = api.search_tweets(q=searchQuery, count=tweetsPerQry, since_id=sinceId, tweet_mode = 'extended')
            else:
                if (not sinceId):
                    new_tweets = api.search_tweets(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1), tweet_mode = 'extended')
                else:
                    new_tweets = api.search_tweets(q=searchQuery, count=tweetsPerQry, max_id=str(max_id - 1), since_id=sinceId, tweet_mode = 'extended')
            if not new_tweets:
                print("No more tweets found")
                break
            all_tweets += new_tweets                
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break
        pass
    print("Downloaded {0} tweets".format(tweetCount))
    return all_tweets

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    return sid_obj.polarity_scores(sentence)

def get_data(all_tweets):
    time_now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("-", "_")
    row = [
        "neg", 
        "neu", 
        "pos", 
        "compound", 
        "tweet_user", 
        "created_at", 
        "tweet_id", 
        "mention_user", 
        "tweet",
        "followers_count"
    ]
    rows = []
    for tweet in all_tweets:
        scores = sentiment_scores(tweet.full_text)
        followers_count = tweet.user.followers_count
        tweet_user = tweet.user.screen_name.lower()
        created_at = tweet.created_at
        tweet_id = tweet.id
        for mention in tweet.entities['user_mentions']:
            mention_user = mention['screen_name'].lower()
            rows.append([
                scores["neg"],
                scores["neu"],
                scores["pos"],
                scores["compound"],
                tweet_user,
                created_at,
                tweet_id,
                mention_user,
                tweet.full_text.encode('ascii',errors='ignore').decode(),
                followers_count
            ])  
    return pd.DataFrame(rows, columns=row)  

def store_data(all_tweets):
    time_now = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_").replace("-", "_")
    row = [
        "neg", 
        "neu", 
        "pos", 
        "compound", 
        "tweet_user", 
        "created_at", 
        "tweet_id", 
        "mention_user", 
        "tweet",
        "followers_count"
    ]
    with open(f"./tweets_{time_now}.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        for tweet in all_tweets:

            scores = sentiment_scores(tweet.full_text)
            followers_count = tweet.user.followers_count
            tweet_user = tweet.user.screen_name.lower()
            created_at = tweet.created_at
            tweet_id = tweet.id
            for mention in tweet.entities['user_mentions']:
                mention_user = mention['screen_name'].lower()
                writer.writerow([
                    scores["neg"],
                    scores["neu"],
                    scores["pos"],
                    scores["compound"],
                    tweet_user,
                    created_at,
                    tweet_id,
                    mention_user,
                    tweet.full_text.encode('ascii',errors='ignore').decode(),
                    followers_count
                ])    

def build_graph(all_tweets, term, positive=True):
    G = nx.MultiDiGraph()
    for tweet in all_tweets:
        if tweet.user.followers_count > 10000:
            scores = sentiment_scores(tweet.full_text)
            scores["followers_count"] = tweet.user.followers_count
            if (positive and scores['compound'] > 0) or (not positive and scores['compound'] < 0):
                for mention in tweet.entities['user_mentions']:
                    user = tweet.user.screen_name.lower()
                    mention_user = mention['screen_name'].lower()
                    G.add_edge(user,mention_user,**copy.copy(scores))
    
    H = nx.Graph(copy.deepcopy(G))
    edgewidth = []
    for u, v in H.edges():
        try:
            edgewidth.append(len(G.get_edge_data(u, v)))
        except:
            edgewidth.append(len(G.get_edge_data(v, u)))

    nodesize = []
    v = term
    for u in H.nodes():
        if u == v:
            nodesize.append(1)
        else:
            data = G.get_edge_data(u, v)
            if data is None:
                data = G.get_edge_data(v, u)
            try:
                average = sum([x['followers_count'] for x in data.values()])/len([x['followers_count'] for x in data.values()])
                nodesize.append(average)
            except:
                nodesize.append(.1)

    pos = nx.kamada_kawai_layout(H)
    fig, ax = plt.subplots(figsize=(100, 100))
    nx.draw_networkx_edges(H, pos, alpha=0.3, width=edgewidth, edge_color="m")

    nx.draw_networkx_nodes(H, pos, node_size=nodesize, node_color="#210070", alpha=0.9)
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(H, pos, font_size=14, bbox=label_options)

    # Title/legend
    font = {"fontname": "Helvetica", "color": "k", "fontweight": "bold", "fontsize": 14}
    if positive:
        ax.set_title("Positive Vader Scored Tweets", font)
    else:
        ax.set_title("Negative Vader Scored Tweets", font)
    # Change font color for legend
    font["color"] = "r"

    ax.text(0.80,0.10,"edge width = # of mentions",horizontalalignment="center",transform=ax.transAxes,fontdict=font,)
    ax.text(0.80,0.06,"node size = number of followers",horizontalalignment="center",transform=ax.transAxes,fontdict=font,)

    # Resize figure for label readibility
    ax.margins(0.1, 0.05)
    fig.tight_layout()
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 7:
        consumer_key=sys.argv[1]
        consumer_secret=sys.argv[2]
        search=sys.argv[3]
        mention=sys.argv[4]
        maxTweets=int(sys.argv[5])
        clusters=int(sys.argv[6])

        all_tweets_list = collect_data(f"@{search}", maxTweets, consumer_key, consumer_secret)
        all_tweets = get_data(all_tweets_list)
        all_tweets = preprocess(all_tweets)
        build_word_cloud(all_tweets)
        dd = lda(all_tweets, clusters)

        # build_graph(all_tweets_list, mention)
        # build_graph(all_tweets_list, mention, positive=False)
        store_data(all_tweets_list)

    else:
        print("you must specify a consumer_key, consumer_secret, mention, maxTweets")


