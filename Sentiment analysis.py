import nltk
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# function to get the synset of a word based on its part of speech
def get_synset(word, pos):
    if pos.startswith('J'):
        # If the word is an adjective
        pos = wordnet.ADJ
    elif pos.startswith('V'):
        # If the word is a verb
        pos = wordnet.VERB
    elif pos.startswith('N'):
        # If the word is a noun
        pos = wordnet.NOUN
    elif pos.startswith('R'):
        # If the word is an adverb
        pos = wordnet.ADV
    else:
        # If the part of speech is unknown, assume it's a noun
        pos = wordnet.NOUN
    # Get the synset of the word based on its part of speech
    synsets = wordnet.synsets(word, pos=pos)
    if synsets:
        return synsets[0]
    else:
        return None

# function to get the sentiment score of a sentence using WordNet and VADER
def get_sentiment(sentence):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    # Get the part of speech for each token
    pos_tags = nltk.pos_tag(tokens)
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    # Calculate the sentiment score using VADER
    vader_score = sid.polarity_scores(sentence)['compound']
    # Calculate the sentiment score using WordNet
    wordnet_score = 0.0
    count = 0
    for word, pos in pos_tags:
        synset = get_synset(word, pos)
        if synset:
            synset_score = sid.polarity_scores(synset.definition())['compound']
            if synset_score != 0:
                wordnet_score += synset_score
                count += 1
    if count != 0:
        wordnet_score /= count
    # Return the average of the VADER and WordNet sentiment scores
    return (vader_score + wordnet_score) / 2

# To Test the sentiment analysis function on some example sentences

a=int(input("Enter 1 for CONSOL and 2 for FILE"))
if(a==1):
  sentences = [i for i in input().split(".")]
else:
  from google.colab import files
  files.upload()
  text = open('read.txt', encoding='utf-8').read()
  sentences=text.split(".")

for sentence in sentences:
    sentiment = get_sentiment(sentence)
    print(f'Sentence: {sentence}\nSentiment Score: {sentiment}\n')
