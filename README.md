# st-RATP_Report
Welcome to the RATP report !
-------------------------------------------------------------------------------

The best dashboard that can guide you for your analysis on the tweets related to RATP.
This dashboard will show you the analysis done on the tweets since the last day and classify them by feelings and topics.

/!\Very important note: When you are in the machine learning section, an embedding step will be triggered and this step is quite long (it can last between 40min and 1h30). You will have to let the page load itself.
![image](https://user-images.githubusercontent.com/73355151/197520347-e54bd58f-a764-444b-af74-fb380c88cf42.png)



To start with, you will need to install some libraries:
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

For this you must install 
---------------------------------------------------------------------

pip install missingno

pip install hdbscan

pip install umap-learn

pip install sentence-transformers

pip3 install git+https://github.com/huggingface/transformers

