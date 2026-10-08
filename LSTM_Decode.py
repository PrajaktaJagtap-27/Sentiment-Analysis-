#Step 1 : import the required libraries
from tensorflow.keras.datasets import imdb
from  tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

###############################################################################
#Step 2 : Configuration of Values
###############################################################################

VOCAB_SIZE = 10000  # Consider most frequent 10,000 unique words
MAX_LEN = 200  # Consider maximum 200 words in each review

###############################################################################
#Step 3 : Load the IMDB dataset
###############################################################################
print("-"*40)
print("Movie Review Sentiment Analysis using LSTM")
print("-"*40)

print("Loading the IMDB dataset...")

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=VOCAB_SIZE)

print("IMDB Dataset loaded successfully!")

print("Number of training reviews :",len(X_train))
print("Number of testing reviews :",len(X_test))

###############################################################################
#  X_train        Review used for tranining
#  Y_train        Actual sentiment of the review 
#  X_test         Review used for testing   
#  Y_test         Actual sentiment  of testing 

#Sentiment :
#  0 : Negative
#  1 : Positive
###############################################################################


###############################################################################
#Step 4 : load the word dictionary
###############################################################################

word_index = imdb.get_word_index()

#Dictionary contains the mapping of words and its corresponding number.
#drisham is good movie  -> (20,56,78,43)
#20 -> drisham
#56 -> is
#78 -> good
#43 -> movie

###############################################################################
#Step 5 : create the reverse  dictionary
###############################################################################

reverse_word_index = {}

for word,index in word_index.items():
    reverse_word_index[index+3] = word

###############################################################################
#Step 6 : Function to decade the review (number to word)
###############################################################################

def DecodeReview(encode_review):
    words = []

    for number in encode_review:
        if number >= 3: #ignor first 3
            word = reverse_word_index.get(number,"?")
            words.append(word)
    return " ".join(words)  #join the list of words

###############################################################################
#Step 7 : Dispaly simple review
###############################################################################
print("-"*40)
print("------------Sample Review------------")
print("-"*40)

for i in range(4):
    review =DecodeReview(X_train[i])

    print("-"*40)

    print("Review number :",i+1)
    print("Review :")
    print(review)

    print("-"*40) 
    
    if Y_train[i] == 1:
        print("Sentiment : POSITIVE")
    else:
        print("Sentiment : NEGATIVE")
