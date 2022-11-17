import nltk.classify.util 
import pandas as pd
import numpy as np
from collections import Counter

#   #   #   # 
#   #   #   #
#   TASK 1  #
#   #   #   #
#   #   #   #  

def data_get_and_clean():
    # Read data from excel file
    movie_data_excel_file = pd.read_excel('movie_reviews.xlsx')
    # Allocate data into a dataframe
    movie_data_dataframe = df = pd.DataFrame(movie_data_excel_file, columns=['Review', 'Sentiment', "Split"])

    # Split dataframe into test and training data
    dataframe_training_data = movie_data_dataframe[(movie_data_dataframe.Split == "train")]
    dataframe_test_data = movie_data_dataframe[(movie_data_dataframe.Split == "test")]

    # Get list of training data reviews and sentiments
    training_data_reviews_list = movie_data_excel_file["Review"][movie_data_excel_file["Split"] == "train"]
    training_data_sentiment_list = movie_data_excel_file["Sentiment"][movie_data_excel_file["Split"] == "train"]

    # Get list of testing data reviews and sentiments
    testing_data_reviews_list = movie_data_excel_file["Review"][movie_data_excel_file["Split"] == "test"]
    testing_data_sentiment_list = movie_data_excel_file["Sentiment"][movie_data_excel_file["Split"] == "test"]

    # Get list of training data negative & positive reviews
    training_data_reviews_list_negative = dataframe_training_data[dataframe_training_data['Sentiment']=='negative']
    training_data_reviews_list_positive = dataframe_training_data[dataframe_training_data['Sentiment']=='positive']
    # Get list of testing data negative & positive reviews
    testing_data_reviews_list_negative = dataframe_test_data[dataframe_test_data['Sentiment']=='negative']
    testing_data_reviews_list_positive = dataframe_test_data[dataframe_test_data['Sentiment']=='positive']


    print("Training Set - Negative Reviews number: " )
    training_data_reviews_number_negative = len(training_data_reviews_list_negative) 
    print(training_data_reviews_number_negative)

    print("Training Set - Positive Reviews number: " )
    training_data_reviews_number_positive = len(training_data_reviews_list_positive)
    print( training_data_reviews_number_positive )

    print("Test Set - Negative Reviews number: ")
    testing_data_reviews_number_negative = len(testing_data_reviews_list_negative) 
    print(testing_data_reviews_number_negative)

    print("Test Set - Positive Reviews number: "   )
    testing_data_reviews_number_positive = len(testing_data_reviews_list_positive) 
    print(testing_data_reviews_number_positive)

    return  training_data_reviews_number_negative, training_data_reviews_number_positive, training_data_reviews_list_negative, training_data_reviews_list_positive, testing_data_reviews_list_negative, testing_data_reviews_list_positive, dataframe_test_data, dataframe_training_data, training_data_reviews_list, training_data_sentiment_list, testing_data_reviews_list,testing_data_sentiment_list


print (
    """
    #   #   #   #
    #   #   #
    #   
    #   TASK 1 - Execution
    #   
    #   #   #
    #   #   #   #
    """
)

training_data_reviews_number_negative, training_data_reviews_number_positive, training_data_reviews_list_negative, training_data_reviews_list_positive, testing_data_reviews_list_negative, testing_data_reviews_list_positive, dataframe_test_data, dataframe_training_data, training_data_reviews_list, training_data_sentiment_list, testing_data_reviews_list,testing_data_sentiment_list = data_get_and_clean()

#   #   #   # 
#   #   #   #
#   TASK 2  #
#   #   #   #
#   #   #   #  

def training_data_reviews_convert_to_words(dataset, min_occurence, min_length):
    # Get a dataset + clean data + remove all special chars + split into words
    dataset['Review'] = dataset['Review'].str.replace('[^a-zA-Z]', ' ')
    dataset_with_reviews = dataset['Review'].str.lower().str.cat(sep=' ')
    reviews_tokenized = nltk.tokenize.word_tokenize(dataset_with_reviews)
    word_dist = nltk.FreqDist(reviews_tokenized)
    
    # Create a dataset with words and frequency of each word
    most_common_words_dataset = pd.DataFrame(word_dist.most_common(), columns=['Word', 'Frequency'])
    most_common_words_dataset_with_requirements = most_common_words_dataset[ (most_common_words_dataset['Frequency'] > min_occurence) & ( (  most_common_words_dataset['Word'].str.len() ) > min_length    ) ]

    # Add data into a dictionary
    dictionary = most_common_words_dataset_with_requirements.set_index('Word')['Frequency'].to_dict()

    return dictionary

print (
    """
    #   #   #   #
    #   #   #
    #   
    #   TASK 2 - Execution
    #   
    #   #   #
    #   #   #   #
    """
)

min_word_occurence = int(input("Define the min. word occurence: "))
min_word_length = int(input("Define the min. word length: "))

# Execute the task 2 function with providing: 1) training data 2) min. word occurence 3) min. word length
words_dictionary = training_data_reviews_convert_to_words(dataframe_training_data, min_word_occurence, min_word_length)
print (words_dictionary )


#   #   #   # 
#   #   #   #
#   TASK 3  #
#   #   #   #
#   #   #   #  

def words_count_occurence_in_reviews(dictionary, dataset):
    # Remove special chars + to lowercase + split into words
    dataset_removed_chars = dataset.str.replace('[^a-zA-Z]', ' ')
    dataset_lowercase = dataset_removed_chars.str.lower()
    dataset_split = dataset_lowercase.str.split()

    # Get all dictionary countable objects 
    # Set a default coount of 0 to each word
    # Counter is a subclass of dict that's specially designed for counting hashable objects
    words_occurence_counter = Counter(dictionary)
    for dictionary_word in words_occurence_counter:
        words_occurence_counter[dictionary_word] = 0

    # We need to run for loop for both index and review, as each review contains an index value
    for index, review in dataset_split.iteritems():
        # If the word is present in the dictionary, increase the counter 
        # update() takes a python iterable object or a mapping object and adds the counts of 
        # elements present in the iterable/mapping object to the counts of elements present in the counter object.
        words_occurence_counter.update(set.intersection(set(review), dictionary))

    return words_occurence_counter


print (
    """
    #   #   #   #
    #   #   #
    #   
    #   TASK 3 - Execution
    #   
    #   #   #
    #   #   #   #
    """
)

movie_data_excel_file = pd.read_excel('movie_reviews.xlsx')
reviews_training_data_list = movie_data_excel_file[movie_data_excel_file["Split"] == "train"]

print ("Positive Words Occurence Dictionary /w Count")
dataset = reviews_training_data_list["Review"][reviews_training_data_list['Sentiment']=='positive']
words_occurence_count_positive = words_count_occurence_in_reviews(words_dictionary, dataset)
print (words_occurence_count_positive)

print ("Negative Words Occurence Dictionary /w Count")
dataset = reviews_training_data_list["Review"][reviews_training_data_list['Sentiment']=='negative']
words_occurence_count_negative = words_count_occurence_in_reviews(words_dictionary, dataset)
print (words_occurence_count_negative)


#   #   #   # 
#   #   #   #
#   TASK 4  #
#   #   #   #
#   #   #   #  

def likelihood_using_laplace(positive_words_frequency, negative_words_frequency, total_reviews_positive, total_reviews_negative):
    # words likelihood dictionary
    likelihood_dictionary = {}  

    # Define a value of alpha smoothing
    alpha_smoothing_laplace = 1
    prior_strength = 1

    # Go through each word from the words frequency
    # Doesn't matter if we run a for loop for positive or negative words as both sets have the same size
    for word in positive_words_frequency:
        # Calculate both positive and negative likelihood of the words occurence using laplace
        likelihood_positive_laplace = (positive_words_frequency.get(word) + alpha_smoothing_laplace) / (total_reviews_positive + prior_strength * alpha_smoothing_laplace)
        likelihood_negative_laplace = (negative_words_frequency.get(word) + alpha_smoothing_laplace) / (total_reviews_negative + prior_strength * alpha_smoothing_laplace)

        # Merge positive and negative likelihood into a single object
        likelihood_ratio_positive_negative = [likelihood_positive_laplace, likelihood_negative_laplace]
        # Add a likelihood ratio object to the dictionary
        likelihood_dictionary[word] = likelihood_ratio_positive_negative

    # Calculate the positive reviews ratio in the context of all reviews
    positive_reviews_in_total_reviews_ratio = (total_reviews_positive) /  (total_reviews_positive + total_reviews_negative)
    # Calculate the negative reviews ratio in the context of all reviews
    negative_reviews_in_total_reviews_ratio = (total_reviews_negative) /  (total_reviews_positive + total_reviews_negative)

    return likelihood_dictionary, positive_reviews_in_total_reviews_ratio, negative_reviews_in_total_reviews_ratio


print (
    """
    #   #   #   #
    #   #   #
    #   
    #   TASK 4 - Execution
    #   
    #   #   #
    #   #   #   #
    """
)

likelihood_dictionary_laplace, positive_reviews_in_total_reviews_ratio, negative_reviews_in_total_reviews_ratio = likelihood_using_laplace(words_occurence_count_positive, words_occurence_count_negative, training_data_reviews_number_positive, training_data_reviews_number_negative)
print ("Likelihood Dictionary: ")
print (likelihood_dictionary_laplace)
print ("Positive reviews ratio in the context of all reviews: ")
print (positive_reviews_in_total_reviews_ratio)
print ("Negative reviews ratio in the context of all reviews: ")
print (negative_reviews_in_total_reviews_ratio)


#   #   #   # 
#   #   #   #
#   TASK 5  #
#   #   #   #
#   #   #   #  

def max_likelihood_classification(text, positive_reviews_in_total_reviews_ratio, negative_reviews_in_total_reviews_ratio, likelihood_dictionary_laplace):

    # Text clean - Remove special chars + to lowercase + split into words
    text = text.replace('[^a-zA-Z]', ' ')
    text = text.lower()
    text = text.split()

    # Set a default value to positive and negative scores
    positive_score = 0
    negative_score = 0

    # Go through each word in text
    for word in text:
        # If word is in the likelihood dictionary
        if word in likelihood_dictionary_laplace:
            # Add a words positive and negative scores to the likelihood general scores

            # We use np.log here to respond to skewness towards large values; i.e., 
            # cases in which one or a few points are much larger than the bulk of the data

            word_positive_score = np.log (likelihood_dictionary_laplace[word][0])
            positive_score = positive_score + word_positive_score

            word_negative_score = np.log(likelihood_dictionary_laplace[word][1])
            negative_score = negative_score + word_negative_score

    # Check if overall positive score is higher than negative score
    # If it's then it means that the text sentiment is positive
    # np.exp is used to calculate the exponential of all elements in the input array
    if np.exp(positive_score - negative_score) > np.exp(np.log(negative_reviews_in_total_reviews_ratio) - np.log(positive_reviews_in_total_reviews_ratio)):
        # If it's positive - return 1 = Positive
        return 1
    else:
        # If it's positive - return 0 = Negative
        return 0


print (
    """
    #   #   #   #
    #   #   #
    #   
    #   TASK 5 - Execution
    #   
    #   #   #
    #   #   #   #
    """
)

review_example = input("Enter a review: ")
max_likelihood_classification_result = max_likelihood_classification(review_example, positive_reviews_in_total_reviews_ratio, negative_reviews_in_total_reviews_ratio, likelihood_dictionary_laplace)

if (max_likelihood_classification_result == 1):
    print ("Text Sentiment: Positive")
elif (max_likelihood_classification_result == 0):
    print ("Text Sentiment: Negative")


#   #   #   # 
#   #   #   #
#   TASK 6  #
#   #   #   #
#   #   #   #


#   #   #   # 
#   #   #   #
#   TASK 7  #
#   #   #   #
#   #   #   #


def user_review_analysis(review):
    max_likelihood_classification_result = max_likelihood_classification(review, positive_reviews_in_total_reviews_ratio, negative_reviews_in_total_reviews_ratio, likelihood_dictionary_laplace)
    if (max_likelihood_classification_result == 1):
        print ("Text Sentiment: Positive")
    elif (max_likelihood_classification_result == 0):
        print ("Text Sentiment: Negative")

print (
    """
    #   #   #   #
    #   #   #
    #   
    #   TASK 7 - Execution
    #   
    #   #   #
    #   #   #   #
    """
)

while True:
        review = input("Enter your review: ")
        user_review_analysis(review)

    




  