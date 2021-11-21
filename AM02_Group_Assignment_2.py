#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author : YOUR GROUP NUMBER GOES HERE
AM02 Group Assignment 2 
"""

#%% LOAD LIBRARIES
# Ideally create another environment where you will install some of the libraries with pip
# You can also install it to your current spyder2021 env however if a problem occurs you will need to re-create the environment

# Install these in the terminal (or Anacond Prompt if on windows):
'''    
pip install tweet-preprocessor
'''
# If above didn't work for Windows user try:
'''
pip install -i https://pypi.anaconda.org/berber/simple tweet-preprocessor 
'''
# Also install wordcloud
'''
pip install wordcloud
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   # main plotting library
import re # regular expressions (string matching)
# preprocess tweets to remove: URLs, emojies, smileys, hashtags, mentions
import preprocessor as p 
from PIL import Image # Python Imagining Lib (we will import image)
# natural language processing library (nat. lang. toolkit)
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer #sentiment analysis
from nltk.corpus import stopwords # remove meaningless words for wordcloud plot
from wordcloud import WordCloud   


#%% TWEETS PREPROCESSING -----------------------------------------------------:
# Load the data from pickle (these were stored as a dataframe)
df_all = pd.read_pickle("Tesla_Share_Tweets_1Mth")
tweets = list(df_all['text']) # will allow us to use both pandas and numpy fns

# Number of downloaded tweets (approx 3,500)

# First tweet

# Remove duplicate tweets from the list (re-tweets; news repeats)

# --- CLEAN DATA
# Based on which libraries will be used afterwards, some of the below data 
# preprocessing may be needed. 
# Note: the sentiment analysis library handles the below for you, however
# let's understand how to do it in case you need to:

# 1) --- Use preprocessor library (which we called 'p') to clean data from:
# URLs, emojies, smileys, hashtags, mentions 
# (note the simple senitment lib does not recognise emojies, or #hashtags)

# Test string to see how preprocessing library p works:
s = 'Tesla cybertruck is great! #awesome 👍 https://www.tesla.com/en_gb/cybertruck'
ans = p.clean(s)
print(ans) # Tesla cybertruck is great!
print(s)   # remember original string will be unchaned as it is immutable

# Clean all tweets using list comprehension

# Check how the first 5 tweets look like now:


# Clean tweets using regular expressions. 
# Keep only letters and replace everything else with a space. Syntax is:
# re.sub(what you keep, what you repalce it with, string)
tws2 = [re.sub("[^a-zA-Z]", " ", x) for x in tws1] 

# Let's check how the first 5 tweets look like now:


# Strip all tweets of start/end white spaces and place all text to lower case 
# Use string methods to:
# strip characters of all start/end white spaces
# place all text to lower case 

# Let's check how the first 5 tweets look like now:

# Delete any empty tweets (perhaps they only contained emojies)


#%% Q1 II. PERFORM SENTIMENT ANALYSIS -----------------------------------------------:
# a) sentiment analysis
nltk.download('vader_lexicon') # only necessry for Mac machines
# initisalise VADER sentiment analysis lexicon, such that its ready for use
sid = SentimentIntensityAnalyzer() 

'''
Each word in a tweet is analysed for its negative / neutral / positive valence.
Then sentiment analyser produces a 'polarity score', standardized to range 
-1, +1, which gives the overall affect of the entire text.
'''

# Let's test out some strings
score = sid.polarity_scores('i love my awesome car, absolutely beautiful') 
print(score) # 'compound': 0.9259

score = sid.polarity_scores('awesome car i will buy it') 
print(score) # 'compound': 0.6249

score = sid.polarity_scores('not good') 
print(score) # 'compound': -0.3412

score = sid.polarity_scores('something neurtral') 
print(score) # 'compound': 0.0

# b) TASK: WRITE CODE WHICH WOULD PRODUCE AV. 'compound' SCORE FOR TESLA TWEETS




# c) Comment on the final score


#%% WORDCLOUD ----------------------------------------------------------------:
# Obtain words that should not be plotted as they are not of much interest
nltk.download('stopwords') # download stopwords 
sw = set(stopwords.words('english'))
print(sw) # common Engish stopwords 

# Obtain all tweets as a single string to be used inside WordCloud function
allTweets = " ".join(tws4) # join all tweets into single string 

# Make standard Wordcloud
worC = WordCloud(width = 1000, height = 700, margin = 0, max_font_size = 170, min_font_size = 25, stopwords = sw).generate(allTweets)
plt.figure(figsize=(8, 6))
plt.imshow(worC, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

# Make a wordcloud in a shape of the Tesla Cybertruck!
# Open up the Console wide to allow full image to display
wave_mask = np.array(Image.open("cybertruck.jpg"))
wordcloud = WordCloud(width = 1000, height = 700, max_font_size = 50, min_font_size = 5, mask = wave_mask).generate(allTweets)
plt.figure(figsize=(14, 12))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


#%% Q2 FED RESERVE STATEMENT -------------------------------------------------:
# September statement's important sentence phrasing
fomc_sep2018 = ["the", "committee", "expects", "that", "further", "gradual", "increases", "in", "the", "target", "range"]
# December statement's important sentence phrasing
fomc_dec2018 = ["the", "committee", "judges", "that", "some", "further", "gradual", "increases", "in", "the", "target", "range"]
   




















