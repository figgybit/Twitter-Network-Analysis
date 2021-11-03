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

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def collect_data(mention, maxTweets, consumer_key, consumer_secret):
    auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
    auth.secure = True
    api = tweepy.API(auth)
    searchQuery = mention
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
    print ("Downloaded {0} tweets".format(tweetCount))
    return all_tweets

def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    return sid_obj.polarity_scores(sentence)

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

def build_graph(all_tweets, positive=True):
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
    v = 'meta'
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
    if len(sys.argv) == 5:
        consumer_key=sys.argv[1]
        consumer_secret=sys.argv[2]
        mention=sys.argv[3]
        maxTweets=int(sys.argv[4])
        all_tweets = collect_data(mention, maxTweets, consumer_key, consumer_secret)
        build_graph(all_tweets)
        build_graph(all_tweets, positive=False)
        store_data(all_tweets)

    else:
        print("you must specify a consumer_key, consumer_secret, mention, maxTweets")


