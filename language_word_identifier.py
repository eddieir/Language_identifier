from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder, BigramAssocMeasures
from nltk import ngrams, NgramAssocMeasures
import re
import os
import math
import numpy as np
import string
import argparse

global DEBUG
DEBUG = False
MODELS_PATH = "models/"
TRAIN_PATH = "training/"
FREQ_FILTER = 1

# Remove anything that is not lowercase and uppercase letters
def pre_processing(line):
    line = re.sub(r'[0-9\'!"#$%&\'()*+,-./:;]','', line).lower()
    return line

# Filter words
def filter_words(training_path, words):
    with open(training_path, 'r') as fpr:
        for i, row in enumerate(fpr):
            row = pre_processing(row.strip())
            words.append(row)
            words.append(' ')

# Training language models
def train_language(language, training_path):
    words = []
    filter_words(training_path, words)
    seq = ' ' + ''.join(words)

    # Bigram
    bigram_finder = BigramCollocationFinder.from_words(seq)
    bigram_finder.apply_freq_filter(FREQ_FILTER)
    bigram_model = bigram_finder.ngram_fd.items()

    # Trigram
    trigram_finder = TrigramCollocationFinder.from_words(seq)
    trigram_finder.apply_freq_filter(FREQ_FILTER)
    trigram_model = trigram_finder.ngram_fd.items()

    # Quad
    quadgram_finder = QuadgramCollocationFinder.from_words(seq)
    quadgram_finder.apply_freq_filter(FREQ_FILTER)
    quadgram_model = quadgram_finder.ngram_fd.items()

    bigram_model = sorted(bigram_finder.ngram_fd.items(), key=lambda item: item[1], reverse=True)
    trigram_model = sorted(trigram_finder.ngram_fd.items(), key=lambda item: item[1], reverse=True)
    quadgram_model = sorted(quadgram_finder.ngram_fd.items(), key=lambda item: item[1], reverse=True)
    
   
    final_model = bigram_model + trigram_model + quadgram_model
    #print(final_model)
    np.save(MODELS_PATH+language+'.npy',final_model)
    print("Language model for {} stored at {}".format(language, MODELS_PATH+language+'.npy'))

# Process model to store the result
def analyze_model():
    all_models = os.listdir(MODELS_PATH)
    language_model = []

    for model_file in all_models:
        language_name = re.findall(r'^[a-zA-Z]+', model_file)
        language_data = []
       
        model_dict = dict()
        model = np.load(MODELS_PATH+model_file)
        print("Language:{}\t Number of n-gram: {} ".format(language_name, len(model)))

        for item in model:
            if item[0] not in model_dict: model_dict[item[0]] = item[1]

        language_model.append((model_file, model_dict, len(model)))

    return language_model


def predict(test_string, models):
    # clean string
    test_string = pre_processing(test_string)

    bi_test = BigramCollocationFinder.from_words(test_string)
    tri_test = TrigramCollocationFinder.from_words(test_string)
    quad_test = QuadgramCollocationFinder.from_words(test_string)
    final_test = list(bi_test.ngram_fd.items()) + list(tri_test.ngram_fd.items()) + list(quad_test.ngram_fd.items())
   
    model_name = []

    for model in models:
        model_name.append(model[0])

    freq_sum = np.zeros(len(models))
    for ngram, freq in final_test:
        exists = 0

        for i, lang_model in enumerate(models):
            lang = lang_model[0]
            model = lang_model[1]
            total_ngram = lang_model[2]
            
            if ngram in model:
                if DEBUG: print("Found", ngram, model[ngram], lang, total_ngram)
                # normalizing to prevent freq/total to be zero
                freq_sum[i] = freq_sum[i] + (freq*10000)/total_ngram
                exist = 1

            if not exists:
                freq_sum[i] += 1

        max_val = freq_sum.max()
        index = freq_sum.argmax()

    if not max(freq_sum):
        if DEBUG: print("[ERROR] Invalid string. String: {}".format(test_string))
        return 0, "Hmm, I do not know this word. Please try other words."

    # get highest score and normalize it to be between 0,1}
    _max = 0
    freq_to_model = list(zip(freq_sum, model_name))
    scores = [x for x, y in freq_to_model]
    normalized_scores_name = [ (normalize_score(f, scores), m) for f, m in freq_to_model ]
    sorted_score_model = sorted(normalized_scores_name, reverse=True)
   
    if DEBUG: print("[DEBUG] Frequency to model: {}".format(freq_to_model))
    if DEBUG: print("[DEBUG] Scores: {}".format(scores))
    if DEBUG: print("[DEBUG] Normalized scores name: {}".format(normalized_scores_name))
    if DEBUG: print("[DEBUG] Reverse sorted score model: {}".format(sorted_score_model))

    return 1, sorted_score_model

def normalize_score(x, score):
    return (x-min(score))/(max(score) - min(score))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Identify language of given string")
    sub_parsers = parser.add_subparsers(help="help for subcommands", dest="mode")

    # train language arguments
    train_parser = sub_parsers.add_parser('train', help='train commands')
    train_parser.add_argument("-i", "--input", help="Training directory", required=True)

    # predict language arugments
    predict_parser = sub_parsers.add_parser('predict', help='predict commands')
    predict_parser.add_argument("-d", help="Debug messages on", action='store_true')

    return parser.parse_args()

def get_filepath(path):
    file_info = []
    if os.path.isdir(path):
        for file_path in os.listdir(path):
            name = re.findall(r'^[a-z]+', file_path).pop()
            file_info.append((name,TRAIN_PATH + file_path))

    return file_info

if __name__ == "__main__":
    args = parse_arguments()

    if args.mode == "train":
        pair_lang_train_path = get_filepath(args.input)
        for lang, train_path in pair_lang_train_path:
            print("Training language {}".format(lang))
            train_language(lang, train_path)

    elif args.mode == "predict":
        if args.d:
            DEBUG=True
            print("Debug on")

        models = analyze_model()
        print("Predicting words (type DONE to quit):")
        while True:
            input_string = input("What to predict? > ")
            if input_string == "DONE":
                break
            else:
                status, predicted_model = predict(input_string, models)
                if status:
                    top_score = re.findall(r'[0-9]+.[0-9]{1,2}', str(predicted_model[0][0]))
                    top_name = re.findall(r'^[a-z]+', str(predicted_model[0][1]))
                    second_score = re.findall(r'[0-9]+.[0-9]{1,2}', str(predicted_model[1][0]))
                    second_name = re.findall(r'^[a-z]+', str(predicted_model[1][1]))
                else:
                    print(predicted_model)
                    continue

            print('Predicting: {}\t[Guessed: {}, Score: {}][Possible: {}, Score: {}]'.format(
                input_string, top_name[0], top_score[0], second_name[0], second_score[0]))
        print("Goodbye.")