import os
import numpy as np
import sys
from keras.preprocessing.sequence import pad_sequences
import re
import json

def generate_data(GLOVE_DIMENSION=25, MAX_WORDS=40, FULL=False):
    """This method does the preprocessing of the tweets and generates all the data.

    Attributes
    ----------
    GLOVE_DIMENSION : int
        Integer number representing which GloVe embedding to be used, must be 25, 50, 100 or 200.
    MAX_WORDS : int
        The length of the tweet, in our case the number of word vectors every tweet should contain.
    FULL : boolean
        Whether to use full or sample dataset.

    """
    # Loading the constants
    with open("config/constants.json") as f:
        CONSTANTS = json.load(f)
    
    # Loading the data
    if FULL:
        train_pos_file = open(CONSTANTS["train_pos_full"], 'r', encoding='UTF-8')
        train_neg_file = open(CONSTANTS["train_neg_full"], 'r', encoding='UTF-8')
    else:
        train_pos_file = open(CONSTANTS["train_pos"], 'r', encoding='UTF-8')
        train_neg_file = open(CONSTANTS["train_neg"], 'r', encoding='UTF-8')
    test_file = open('data/twitter-datasets/test_data.txt', 'r', encoding='UTF-8')
    
    print('Reading and processing tweets...')
    train_tweets = []  
    test_tweets = []  
    labels = []
    
    # Reading the positive tweets
    for line in train_pos_file:
        tweet = preprocess_tweet(line)
        train_tweets.append(tweet)
        labels.append(1)
    train_pos_file.close()
    print('Positive tweets proccessed.')
    
    # Reading the negative tweets
    for line in train_neg_file:
        tweet = preprocess_tweet(line)
        train_tweets.append(tweet)
        labels.append(0)
    train_neg_file.close()
    print('Negative tweets proccessed.')
    
    # Preprocessing the tweets
    for line in test_file:
        tweet = preprocess_tweet(line)
        test_tweets.append(tweet)
    test_file.close()
    print('Test tweets proccessed.')
    print('Total of %s train tweets.' % len(train_tweets))
    print('Total of %s test tweets.' % len(test_tweets))
    
    # Mapping every unique word to a integer (bulding the vocabulary)
    print('Bulding the vocabulary...')
    word_to_index = {}
    words_freq = {}
    m = 0

    for i, tweet in enumerate(train_tweets):
        words = tweet.split()

        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                word_to_index[word] = m
                m += 1
            if word not in words_freq:
                words_freq[word] = 1
            else:
                words_freq[word] += 1

    word_to_index["unk"] = m
    vocabulary_size = len(word_to_index)
    print('Bulding the vocabulary done, vocabulary size: %s.' % vocabulary_size)
    
    print('Converting training tweets to integer sequences...')
    train_sequences = []

    for i, tweet in enumerate(train_tweets):
        words = tweet.split()

        tweet_seq = []
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                tweet_seq.append(word_to_index["unk"])
            else:
                tweet_seq.append(word_to_index[word])

        train_sequences.append(tweet_seq)

    # Padding the sequences to match the `MAX_WORDS`
    X_train = pad_sequences(train_sequences, maxlen=MAX_WORDS, padding="post", value=vocabulary_size)
    print('Conversion done.')
    print(X_train.shape)
    
    print('Converting testing tweets to integer sequences...')
    test_sequences = []

    for i, tweet in enumerate(test_tweets):
        words = tweet.split()

        tweet_seq = []
        for word in words[:MAX_WORDS]:
            if word not in word_to_index:
                tweet_seq.append(word_to_index["unk"])
            else:
                tweet_seq.append(word_to_index[word])

        test_sequences.append(tweet_seq)
    
    # Padding the sequences to match the `MAX_WORDS`
    X_test = pad_sequences(test_sequences, maxlen=MAX_WORDS, padding="post", value=vocabulary_size)
    print('Conversion done.')
    print(X_test.shape)
    
    print('Reading glove embeddings...')
    glove_embeddings_path = CONSTANTS["glove_folder"] + '/glove.twitter.27B.' + str(GLOVE_DIMENSION) + 'd.txt'
    glove_embeddings_file = open(glove_embeddings_path, 'r', encoding='UTF-8')

    glove_embeddings = dict()
    for line in glove_embeddings_file:
        parts = line.split()
        key = parts[0]
        embedding = [float(t) for t in parts[1:]]
        glove_embeddings[key] = np.array(embedding)
    print ("Done reading embeddings")
    
    # Generating the embedding matrix for our vocabulary (this is needed for the Embedding layer in keras models)
    print('Generating the embedding matrix...')
    unknowwn = []
    hits = 0
    embedding_matrix = np.zeros((vocabulary_size + 1, GLOVE_DIMENSION))
    for word, idx in word_to_index.items():
        if word in glove_embeddings:
            emb = glove_embeddings[word]
            embedding_matrix[idx] = emb
            hits += 1
        else:
            unknowwn.append(word)
            emb = glove_embeddings["unk"]
            embedding_matrix[idx] = emb

    embedding_matrix[vocabulary_size] = [0]*GLOVE_DIMENSION
    print('Generating done.')
    print('%s words of %s found' % (hits, vocabulary_size))
    print(embedding_matrix.shape)
    
    print('Saving everything...')
    dir_name = "data/glove_%s_words_%s_full_%s" % (GLOVE_DIMENSION, MAX_WORDS, FULL)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    np.save(os.path.join(dir_name, "embedding_matrix"), embedding_matrix)
    np.save(os.path.join(dir_name, "X_train"), X_train)
    np.save(os.path.join(dir_name, "Y_train"), labels)
    np.save(os.path.join(dir_name, "X_test"), X_test)

    print("Saving done.")
    
    
def preprocess_tweet(tweet):
    # MULTILINE - '^' matches the beggining of each line
    # DOTALL - '.' matches every character including newline
    FLAGS = re.MULTILINE | re.DOTALL
    
    # Replace links with token <url>
    tweet = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ", tweet, flags = FLAGS)
    
    # Eyes of a smiley can be represented with: 8:=;
    # Nose of a smiley can be represented with: '`\-
    
    # Replace smiling face with <smile>. Mouth can be repredented with: )dD.
    tweet = re.sub(r"[8:=;]['`\-]?[)dD]+|[(dD]+['`\-]?[8:=;]", " <smile> ", tweet, flags = FLAGS)
    
    # Replace lol face with <lolface>. Mouth can be repredented with: pP
    tweet = re.sub(r"[8:=;]['`\-]?[pP]+", " <lolface> ", tweet, flags = FLAGS)
    
    # Replace sad face with <sadface>. Mouth can be repredented with: (
    tweet = re.sub(r"[8:=;]['`\-]?[(]+|[)]+['`\-]?[8:=;]", " <sadface> ", tweet, flags = FLAGS)
    
    # Replace neutral face with <neutralface>. Mouth can be repredented with: \/|l
    tweet = re.sub(r"[8:=;]['`\-]?[\/|l]+", " <neutralface>", tweet, flags = FLAGS)
    
    # Split concatenated words wih /. Ex. Good/Bad -> Good Bad
    tweet = re.sub(r"/"," / ", tweet, flags = FLAGS)
    
    # Replace <3 with <heart>
    tweet = re.sub(r"<3"," <heart> ", tweet, flags = FLAGS)
    
    # Replace numbers with <number>.
    tweet = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", tweet, flags = FLAGS)
    
    # Replace repeated punctuation with <repeat>. Ex. !!! -> ! <repeat>
    tweet = re.sub(r"([!?.]){2,}", r"\1 <repeat> ", tweet, flags = FLAGS)
    
    # Replace elongated endings with <elong>. Ex. happyyy -> happy <elong>
    tweet = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong> ", tweet, flags = FLAGS)
    
    # Replace multiple empty spaces with one
    tweet = re.sub('\s+', ' ', tweet, flags = FLAGS)
    
    # Remove apostrophes
    tweet = re.sub(r"'","", tweet, flags = FLAGS)
    
    # Return result
    return tweet
