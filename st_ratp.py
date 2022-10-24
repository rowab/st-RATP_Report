from sre_constants import NEGATE
from urllib import response
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import streamlit.components.v1 as components
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from tqdm import tqdm
import plotly.express as px

import tweepy
from langdetect import detect
from textblob import TextBlob
from textblob.compat import PY2, request, urlencode
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
from wordcloud import WordCloud, STOPWORDS
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from sentence_transformers import SentenceTransformer
import hdbscan
import umap

from sklearn.decomposition import PCA
import plotly.graph_objs as go
from sklearn.cluster import KMeans 

def display_tweet(tweet_url):
    api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
    response= requests.get(api)
    res= response.json()["html"]

    return res

def display(input):
    res=display_tweet(input)
    components.html(res,height=700)

def home():
    input1="https://twitter.com/PatrickAdemo/status/1580594236721725446"
    input="https://twitter.com/VictoireValent1/status/1583321236339585027"
    input2="https://twitter.com/faureolivier/status/1583142538558701568"
    input3="https://twitter.com/TriquiBadroudin/status/1582070758536142848"
    col1, col2, col3 = st.columns([5,5,5])
    with col2:
        image = Image.open('RATP.svg.png')
        st.image(image)
    display(input)
    display(input3)
    display(input2)
    display(input1)
    st.markdown("As you will have seen **RATP** has been doing it for a long time until being in TOP tweet sometimes for more than 24h 🐦 !")
    st.markdown("Transportation is an integral part of our lives and today its technology continues to evolve. The RATP group is sought after throughout the world for its unique experience in the operation and maintenance of different modes of transport, as well as in project design and project management. to ensure its development, ratp takes a close look at the opinions of its passengers.")
    st.markdown("The RATP, as mentioned above, is a world-renowned company and is even sought after worldwide for its experience. RATP exports the Group's operating and maintenance know-how on all modes of transport to new territories on a regular basis. However, the company has often been criticized for increasing absenteeism, recruitment difficulties and opening up to competition.")
    st.markdown("It is interesting for us today to ask ourselves what the majority of people really think of this transport company and to see if opinions vary from one place to another.") 
    st.markdown("**Does RATP have such a good reputation?**")
    st.markdown("**Are companies around the world right to take their cues from them and ask for help?**")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("**_Project proposed by: Rowa Bedewy, Hugo Grandjean, Farshad Shamlu_**")

##################################################### Data preprocessing #####################################################################


@st.experimental_memo
def load_dataset():
    DATA_DIR = Path('twitter_data1')
    rows = []
    for file_path in tqdm(DATA_DIR.iterdir()):
        if file_path.is_dir():
            continue
        d = read_json(file_path)
        
        rows.append(dict(
            date_time = d['created_at'],
            name = d['user']['name'],
            followers = d['user']['followers_count'],
            following = d['user']['friends_count'],
            follower_following_ratio =  d['user']['followers_count'] / (d['user']['friends_count'] + 1),
            text = d.get('full_text') or d.get('text'),
            hashtags = list(map(lambda item: item['text'], d['entities']['hashtags'])),
            likes = d['favorite_count'],
            retweets = d['retweet_count'],
        ))
    df = pd.DataFrame(rows)
    return df

def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)

def display_table(data, text):
    if st.checkbox(text):
        st.subheader('Raw data')
        st.write(data)

def display_table2(data, text):
    if st.checkbox(text):
        st.subheader('Raw data')
        st.write(data)

@st.experimental_memo
def Detect_language(df):
    result = ['Not Detect']*len(df)
    for i in range(len(df)):
        try:
            result[i] = detect(df['text'][i])
        except:
            pass
    pd.options.mode.chained_assignment = None
    df['lan'] = result
    return df

########################################################### Vizualisation #######################################################

def getPolarity(text):
    if TextBlob(text).sentiment.polarity <= -.1:
        return -1
    if TextBlob(text).sentiment.polarity >= .1:
        return 1
    else:
        return 0 

def sentiment(df):
     df["sent"] = df['text'].apply(getPolarity)
     return df

def getPolarity_letter(text):
    if TextBlob(text).sentiment.polarity <= -.1:
        return "NEGATE"
    if TextBlob(text).sentiment.polarity >= .1:
        return "POSITIVE"
    else:
        return "NEUTRAL" 


def number_sent(df):
    df["sent"] = df['text'].apply(getPolarity_letter)
    select = st.sidebar.markdown('### Distribution of the tweets')
    select = st.sidebar.selectbox('Choose visualization type', ['Bar plot', 'Pie chart'], key='1')
    sentiment_count = df['sent'].value_counts()
    sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
    if not st.sidebar.checkbox("Hide", True):
        st.markdown("#### Number of tweets by sentiment")
        if select == 'Bar plot':
            fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
            st.plotly_chart(fig)
        else:
            fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
            st.plotly_chart(fig)
        st.markdown("The distribution of tweets in neutral can be explained by the fact that at the moment the RATP is announcing a lot of information, like the new CEO or information about the strike. So, a big part of the tweets at the moment are neutral. It follows the negative tweets of people expressing themselves on this new information. ")
    else:
        st.markdown(" 🐦 Press **Hide** to display the visualization of distribution of the number of tweets by sentiment  ")

def random(df):
    df["Sentiment"] = df['text'].apply(getPolarity_letter)
    st.sidebar.markdown("### Select random tweet sentiment")
    random_tweet = st.sidebar.radio("Sentiment", ('NEGATE', 'POSITIVE', 'NEUTRAL'))
    st.markdown("#### The rendom tweet generate ")
    st.success(df.query("Sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0] )   
    st.markdown("")
    st.markdown("")
    st.markdown("")


def cloud(df):
    st.sidebar.markdown("### Word Cloud")
    stopwords = ["https", "co", "RT", "de", 'le', 'et', 'les', 'en', 'la', 't', 'des', 'vous', 'a',
             'un','je','du','i','on','the','il','au','à','sur','dans','est','pas','se','une','ce',
             'que','li','to','it','mon','ou','ne','si','and','nous','tu','moi','of','ils','sont',
             'avec','y','aux','is','par','me','elle','votre','pdg','sncf','l','va','nicolasputsch',
             's','h','ancien','premier','être','fo','d','maxime_boudet',
             'trouveaurelle','pourront','appellent','tête','suite'
            ]
    word_sentiment = st.sidebar.radio('Display word cloud for what Sentiment?', ('NEGATE', 'POSITIVE', 'NEUTRAL'))
    if not st.sidebar.checkbox("Close", True, key='3'):
        st.subheader('Word cloud for %s Sentiment' % (word_sentiment))
        df = df[df['Sentiment']==word_sentiment]
        words = ' '.join(df['text'])
        comment_words = ''
        for val in df.text:
            # typecaste each val to string
            val = str(val)
            tokens = val.split()
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
            
            comment_words += " ".join(tokens)+" "
        wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=800, height=640).generate(comment_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    else:
        st.markdown(" 🐦 Press **Close** to display the visualization of Word cloud by sentiment  ")

    
        
def get_time(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['weekday'] = df['date_time'].dt.weekday
    df['time'] = df['date_time'].dt.hour
    df['weekday'] = df['weekday'].replace(0, 'Monday')
    df['weekday'] = df['weekday'].replace(1, 'Tuesday')
    df['weekday'] = df['weekday'].replace(2, 'Wendsday')
    df['weekday'] = df['weekday'].replace(3, 'Thursday')
    df['weekday'] = df['weekday'].replace(4, 'Friday')
    df['weekday'] = df['weekday'].replace(5, 'Saterday')
    df['weekday'] = df['weekday'].replace(6, 'Sunday')
    st.markdown("#### Representation number of tweets per hour")
    fig= plt.figure(figsize=(14,6))
    sns.countplot(x="time",data=df)
    st.pyplot(fig)
    st.markdown("#### Representation number of tweets per day")
    fig2= plt.figure(figsize=(14,6))
    plt.subplot(212)
    sns.countplot(x= "weekday",data=df)
    st.pyplot(fig2)
################################################# Machine Learning ########################################################


def by_sent(df):

    st.markdown("After analyzing the data, we will now classify the tweets according to whether they are positive, negative or neutral.")
    st.markdown("Classification by sentiment is an important element that will allow you to have a look at what people are saying about RATP. Especially since our model generates a score of:")
    X_train, X_test, y_train, y_test = train_test_split(df[["text"]], df['sent'])
    pipe = make_pipeline(CountVectorizer(), TfidfTransformer())
    pipe.fit(X_train["text"])
    feat_train = pipe.transform(X_train["text"])
    feat_test = pipe.transform(X_test["text"])
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(feat_train, y_train)
    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False
    )
    st.write(clf.score(feat_test, y_test))
    st.markdown(" 🐦 A **near-perfect score** that cannot be overlooked and will help you classify all tweets about RATP as positive, negative or neutral.")

@st.experimental_memo(suppress_st_warning=True)
def bert_sentence(df):
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    embeddings = model.encode(df['text'], show_progress_bar=True)
    umap_embeddings = umap.UMAP(n_neighbors=15,n_components=5,metric='cosine').fit_transform(embeddings)
    cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(umap_embeddings)
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    st.pyplot(fig)
    st.markdown("It is difficult to visualize the individual clusters due to the number of topics generated, more than 100. However, we notice that on the image a structure is still there and we notice clusters")
    st.markdown("🐦 100 topics being too big a value, we applied a model to keep the best topics corresponding to the tweets and thus targeted the comments. Thus we obtain:")
    X = np.array(embeddings)
    pca = PCA(n_components=3)
    pca.fit(X)
    pca_data = pd.DataFrame(pca.transform(X),columns=['FirstComponent','SecondComponent','ThirdComponent'])
    cluster = KMeans(n_clusters=20, random_state=0).fit(pca_data)
    text_data = df.text.to_list()

    cluster = KMeans(n_clusters=20, random_state=0).fit(pca_data)

    docs_df = pd.DataFrame(text_data, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(text_data))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
    data = pd.DataFrame(top_n_words[0][:10])
    data['Class'] = [0]*len(data)
    for i in range(1,len(top_n_words)):
        data_i = data_0 = pd.DataFrame(top_n_words[i][:100])
        data_i['Class'] = [i]*len(data_i)
        data = data.append(data_i)
    data = data.sort_values(by=1,ascending=False)

    st.write(data.drop_duplicates(subset=0).sort_values(by='Class').drop_duplicates(subset='Class',keep='last'))

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    

    return tf_idf, count
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes



def main():

    selected = option_menu(
        menu_title=None,
        options=["Home", "Vizualisation", "MachineLearning"], 
        icons=["house", "book", "envelope"], 
        menu_icon="cast", 
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#116aad", "font-size": "18px"}, 
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#69d1b7"},
        }
    )
    if selected == "Home":
        home()
    if selected == "Vizualisation":
        st.title("Vizualidation of tweets")
        df = load_dataset()
        display_table(df, "Show tweets")
        df_lan=Detect_language(df)
        display_table2(df_lan, "Show tweets with languages")
        random(df_lan)
        number_sent(df_lan)
        cloud(df_lan)
        get_time(df_lan)

    if selected == "MachineLearning":
        st.title('Classification by sentiment')
        df2 = load_dataset()
        df2_lan=Detect_language(df2)
        sentiment(df2_lan)
        by_sent(df2_lan)
        st.title('Classification by topics')
        st.markdown("Other elements, finding the positive and negative tweets is not enough, the most important is to target the requests made by the customers in order to improve or to find the real problem in other cases.")
        bert_sentence(df2_lan)
        
main()
