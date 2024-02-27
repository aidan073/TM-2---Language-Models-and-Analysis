import math
import os
import re
import copy
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from scipy import stats

class unigram:
    """
    A class for setting up a unigram based appearances dictionary for further calculations

    This class contains a method designed to read in song lyrics and calculate the amount
    of appearances for each word in each genre
    """
    def read_files(self, directory_path):
        """
        Read files from the specified directory and count word appearances per genre.

        Args:
            directory_path (str): The path to the directory containing genre-wise files.

        Returns:
            dict: A dictionary containing the number of word appearances per genre.
                    The keys are genre names, and the values are dictionaries where keys
                    are words and values are their respective appearances in the genre.
        """
        dic_genre_word_appearances = {}
        for genre in os.listdir(directory_path): # for genre in lyrics folder
            path = directory_path + "/" + genre

            temp_dic = {} # inner dictionary with key: term value: appearances in the genre
            for file in os.listdir(path): # for song in genre
                with open(path + "/" + file, 'r') as rfile: 
                    for line in rfile: # for line in song
                        token_list = token_prep.get_tokens(line) # tokenize
                        for token in token_list:
                            temp_dic[token] = temp_dic.get(token, 0) + 1 # update frequency for existing term, or add a new term with frequency 1

            dic_genre_word_appearances[genre] = temp_dic # add inner dic to outer dic

        return dic_genre_word_appearances # i.e. {'Blues': {'had': 17, 'a': 56, 'bad': 25}}

class bigram:
    """
    A class for setting up a bigram based appearances dictionary for further calculations

    This class contains a method designed to read in song lyrics and calculate the amount
    of appearances for each word-pair in each genre
    """
    def read_files(self, directory_path):
        """
        Read files from the specified directory and count word-pair appearances per genre.

        Args:
            directory_path (str): The path to the directory containing genre-wise files.

        Returns:
            dic_genre_pair_appearances (dict): A dictionary containing the number of word-pair
            appearances per genre. The keys are genre names, and the values are dictionaries
            where keys are word-pairs and values are their respective appearances in the genre.
        """
        dic_genre_pair_appearances = {}
        for genre in os.listdir(directory_path): # for genre in lyrics folder
            path = directory_path + "/" + genre

            temp_dic = {} # inner dictionary with key: term value: appearances in the genre 
            for file in os.listdir(path): # for song in genre
                with open(path + "/" + file, 'r') as rfile: 
                    for line in rfile: # for line in song              
                        # prepare text for tokenization
                        token_list = token_prep.get_tokens(line) # tokenize
                        token_list.insert(0, "<s>")
                        token_list.append("</s>")
    
                        if not token_list: # if empty line, skip and go to next line
                            continue

                        for i in range(len(token_list) - 1):
                            key = token_list[i] + " " + token_list[i + 1]
                            temp_dic[key] = temp_dic.get(key, 0) + 1 # update frequency for existing term, or add a new term with frequency 1

            dic_genre_pair_appearances[genre] = temp_dic # add inner dic to outer dic

        return dic_genre_pair_appearances # i.e. {'Blues': {'<s> had': 9, ...}}

class mixed:
    """
    A class for a mixed model language model

    This class contains a method, set_lambda, specific to the mixed model
    """
    def set_lambda(self, uni_dic_tfidf, bi_dic_tfidf, directory_path):
        """
        Determines the optimal lambda values for the weighted combination formula. The weighted
        combination formula combines the probabilities of the unigram and bigram language models.
        The optimal lambda values will signify the importance of unigram and bigram in classification
        
        Args:
            uni_dic_tfidf (dict): Dictionary with keys: genres, values: dictionary A. In A, keys: words, values: TF-IDF scores

        Returns:
            best_lambda (float): lambda value that provides best mixed model results
        """
        most_correct_predictions = 0
        best_lambda = 0
        for i in range(11):
            lambda_value = i / 10.0
            curr_correct_predictions = 0
            for file in os.listdir(directory_path): # for song in validation set
                with open(directory_path + "/" + file, 'r') as rfile: 
                    actual_genre = rfile.readline()
                    actual_genre = actual_genre.strip()
                    text = rfile.read()
                    prediction = tester.classify_mixed(text, lambda_value, uni_dic_tfidf, bi_dic_tfidf)
                    if prediction == actual_genre:
                        curr_correct_predictions+=1
            # check if new most_correct_predictions
            if most_correct_predictions < curr_correct_predictions:
                most_correct_predictions = curr_correct_predictions
                best_lambda = lambda_value     

        return best_lambda

class token_prep:
    """
    A class for tokenization
   
    This class contains a method that prepares and tokenizes text
    """
    @staticmethod
    def get_tokens(line):
        """
        Prepares and tokenizes text

        Args:
            line (str): text to tokenize

        Returns:
            token_list (list): list of tokens
        """
        current_line = line.strip()
        current_line = current_line.lower()
        current_line = re.sub(r'[^\w\s]', '', current_line)
        token_list = word_tokenize(current_line)

        return token_list

class tf_idf_calc:
    """
    A class dedicated to calculating TF-IDF

    This class contains separate methods for the calculation of TF, IDF, and TF-IDF
    """
    def get_TF_values(self, dic_genre_word_appearances):
        """
        Calculates term frequency (TF) per token in each genre based on token frequency

        Args:
            dic_genre_word_appearances (dict): genre names as keys and dictionary A as values. In A, keys are the tokens with their
            frequencies as the values

        Returns:
            dic_genre_term_frequency (dict): Dictionary with genre names as keys, and TF-Values as values. These values are also
            a dictionary of tokens as the keys, and their TFs as values
        """
        dic_genre_term_frequency = {}

        for genre, word_appearances in dic_genre_word_appearances.items(): # for genre, inner dictionary (word: # of appearances in genre)
            total_terms = sum(word_appearances.values())
            term_frequency = {word: math.log10((appearances / total_terms)+1) for word, appearances in word_appearances.items()} # new dictionary with words as keys, TF as values
            dic_genre_term_frequency[genre] = term_frequency # set value to tf_per_token dictionary

        return dic_genre_term_frequency


    def get_IDF_values(self, dic_genre_word_appearances):
        """
        Calculates the IDF values for each token

        Args:
            dic_genre_word_appearances (dict): genre name as key and dictionary A as value. In A, keys: tokens values: # of times 
            the tokens appear in a genre

        Returns: 
            dic_idf_values (dict): Dictionary with tokens as the keys and IDF values as values
        """
        dic_idf_values = {}
        num_genres = len(dic_genre_word_appearances.keys())

        for genre in dic_genre_word_appearances:
            for word in dic_genre_word_appearances[genre]: # for words in genre
                dic_idf_values[word] = dic_idf_values.get(word, 0) + 1 # update df for existing word or add a new word with df 1

        for word in dic_idf_values:
            dic_idf_values[word] = math.log10(num_genres / (dic_idf_values[word])) # the values of dic_idf_values are currently df, change them to idf

        return dic_idf_values
    
    def get_TFIDF_values(self, tf, idf):
        """
        Calculates the TF-IDF values for each token

        Args:
            tf (dict): genre names as keys and dictionary A as value. In A, keys: tokens values: TF of the tokens

            idf (dict): genre names as keys and dictionary A as value. In A, key: tokens values: IDF of the tokens

        Returns: 
            dic_genre_tfidf (dict): keys: genres, values: dictionary A. In A, keys: tokens, values: TF-IDF scores
        """
        dic_genre_tfidf = copy.deepcopy(tf)
        for inner_dic in dic_genre_tfidf.values():
            for word in inner_dic:
                inner_dic[word] = inner_dic[word] * idf[word]

        return dic_genre_tfidf

class tester:
    """
    A class for classifying text

    Contains a method for each language model that classifies text
    """
    @staticmethod
    def classify_unigram(tfidf, text, for_mixed):
        """
        Classifies text using a unigram model

        Args:
            tfidf (dict): keys: genres, values: dictionary A. In A, keys: tokens, values: TF-IDF scores

            text (str): text to classify

            for_mixed (bool): if True, will return results dictionary (for mixed-model computation).
            if False, will return genre that best matches the text

        Returns:
            max(results, key=results.get) (str): highest scoring genre in the results dictionary (best match)

            or

            results (dict): keys: genres values: combined TF-IDF scores for the genres
        """
        input = token_prep.get_tokens(text)

        results = {}
        # add up TF-IDF scores for each genre
        for word in input:
            for genre in tfidf:
                if genre in results:
                    results[genre] += tfidf[genre].get(word, 0)
                else:
                    results[genre] = tfidf[genre].get(word, 0)

        if for_mixed == False:
            return max(results, key=results.get)
        else:
            return results
        
    @staticmethod
    def classify_bigram(tfidf, text, for_mixed):
        """
        Classifies text using a bigram model

        Args:
            tfidf (dict): keys: genres, values: dictionary A. In A, keys: tokens, values: TF-IDF scores

            text (str): text to classify

            for_mixed (bool): if True, will return results dictionary (for mixed-model computation).
            if False, will return genre that best matches the text

        Returns:
            max(results, key=results.get) (str): highest scoring genre in the results dictionary (best match)

            or

            results (dict): keys: genres values: combined TF-IDF scores for the genres
        """
        text = text.split('/n')

        results = {}
        # add up TF-IDF scores for each genre
        for line in text:
            line = token_prep.get_tokens(line)
            line.insert(0, "<s>") 
            line.append("</s>")

            for i in range(len(line) - 1):
                for genre in tfidf:
                    curr = line[i] + " " + line[i + 1] # create pairs
                    if genre in results:
                        results[genre] += tfidf[genre].get(curr, 0)
                    else:
                        results[genre] = tfidf[genre].get(curr, 0)

        if for_mixed == False:
            return max(results, key=results.get)
        else:
            return results
        
    @staticmethod
    def classify_mixed(text, lambda_value, uni_dic_tfidf, bi_dic_tfidf):
        """
        Classifies text using a bigram model

        Args:
            text (str): text to classify

            lambda_value (float): optimal lambda value (determined by mixed.set_lambda)

            uni_dic_tfidf (dict): keys: genres, values: dictionary A. In A, keys: tokens, values: TF-IDF scores

            bi_dic_tfidf (dict): keys: genres, values: dictionary A. In A, keys: tokens, values: TF-IDF scores

        Returns:
            max(combined_results, key=combined_results.get) (str): highest scoring genre in combined_results
        """
        unigram_results = tester.classify_unigram(uni_dic_tfidf, text, True)
        bigram_results = tester.classify_bigram(bi_dic_tfidf, text, True)
        combined_results = {}

        # calculate combined results
        for genre in unigram_results:
            combined = lambda_value * unigram_results[genre] + (1 - lambda_value) * bigram_results[genre]
            combined_results[genre] = combined

        return max(combined_results, key=combined_results.get)

class f1_score_calc:
    """
    A class to calculate f1_score

    Provides a method to calculate f1_score
    """
    @staticmethod
    def calculate_f1(predicted_labels, true_labels):
        """
        Calculates f1 score for language models

        Args:
            predicted_labels (list): the predicted classifications provided by a language model

            true_labels (list): the true classifications for what the language model predicted

        Returns:
            weighted_f1 (float): the final f1 score for a lanuage model
        """

        genres = ['Blues', 'Country', 'Metal', 'Pop', 'Rap', 'Rock'] # all possible genres

        # Calculate F1 score for each class
        f1_per_class = {}
        for genre in genres:
            true_binary = [1 if label == genre else 0 for label in true_labels] # Convert true labels to binary for the current genre
            pred_binary = [1 if label == genre else 0 for label in predicted_labels] # Convert predicted labels to binary for the current genre
            f1_per_class[genre] = f1_score(true_binary, pred_binary) # Calculate F1 score for the current genre

        # Convert true and predicted labels to numerical form
        true_labels_numerical = [genres.index(label) for label in true_labels]
        predicted_labels_numerical = [genres.index(label) for label in predicted_labels]
        
        weighted_f1 = f1_score(true_labels_numerical, predicted_labels_numerical, average='weighted') # Calculate weighted average F1 score

        return weighted_f1
    
class evaluator:
    """
    A class to evaluate f1 and significance

    Provides a method to evaluate f1 scores of a model and another method to determine the significance of the results
    """
    def evaluate_f1(self, file_in, uni_dic_tfidf, bi_dic_tfidf, best_lambda):
        """
        Determines the f1 score of each model

        Args:
            file_in (str): direct path of test file

            uni_dic_tfidf (dict): keys: genres values: combined TF-IDF scores for the genres

            bi_dic_tfidf (dict): keys: genres values: combined TF-IDF scores for the genres

            best_lambda (float): lambda value that provides best mixed model results

        Returns:
            uni_f1, bi_f1, mixed_f1 (float): f1 scores of each model
        """
        with open(file_in, 'r') as file:
            next(file) # skip header
            uni_predictions = []
            bi_predictions = []
            mixed_predictions = []
            true_labels = []

            for line in file:
                sections = line.strip().split('\t') # get list with text and genre separated
                uni_predictions.append(tester.classify_unigram(uni_dic_tfidf, sections[1], False))
                bi_predictions.append(tester.classify_bigram(bi_dic_tfidf, sections[1], False))
                mixed_predictions.append(tester.classify_mixed(sections[1], best_lambda, uni_dic_tfidf, bi_dic_tfidf))
                true_labels.append(sections[2])

            uni_f1 = f1_score_calc.calculate_f1(uni_predictions, true_labels)
            bi_f1 = f1_score_calc.calculate_f1(bi_predictions, true_labels)
            mixed_f1 = f1_score_calc.calculate_f1(mixed_predictions, true_labels)
        
        return uni_f1, bi_f1, mixed_f1
    
    def evaluate_significance(self, unigram_scores, bigram_scores, mixed_scores):
        """
        Calculates significance of the results from each model and prints them

        Args:
            unigram_scores (list): list of unigram f1 scores for each run

            bigram_scores (list): list of bigram f1 scores for each run

            mixed_scores (list): list of mixed f1 scores for each run
        
        """
        # Calculate differences between pairs of F1 scores
        diff_unigram_bigram = [unigram_scores[i] - bigram_scores[i] for i in range(len(unigram_scores))]
        diff_unigram_mixed = [unigram_scores[i] - mixed_scores[i] for i in range(len(unigram_scores))]
        diff_bigram_mixed = [bigram_scores[i] - mixed_scores[i] for i in range(len(bigram_scores))]

        # Perform paired t-tests
        t_stat_unigram_bigram, p_value_unigram_bigram = stats.ttest_rel(diff_unigram_bigram, [0] * len(diff_unigram_bigram))
        t_stat_unigram_mixed, p_value_unigram_mixed = stats.ttest_rel(diff_unigram_mixed, [0] * len(diff_unigram_mixed))
        t_stat_bigram_mixed, p_value_bigram_mixed = stats.ttest_rel(diff_bigram_mixed, [0] * len(diff_bigram_mixed))

        # Print results
        print("Paired t-test results:")
        print("Unigram vs. Bigram: t-statistic =", t_stat_unigram_bigram, " p-value =", p_value_unigram_bigram)
        print("Unigram vs. Mixed: t-statistic =", t_stat_unigram_mixed, " p-value =", p_value_unigram_mixed)
        print("Bigram vs. Mixed: t-statistic =", t_stat_bigram_mixed, " p-value =", p_value_bigram_mixed)


def main():
    text = "You used to call me on my cell phone/nLate night when you need my love/nCall me on my cell phone"

    # create instances of objects
    unigram_model = unigram()
    bigram_model = bigram()
    tf_idf_calculations = tf_idf_calc()
    mixed_model = mixed()
    evaluator1 = evaluator()

    # get token counts in genres
    dic_genre_pair_count = bigram_model.read_files("Lyrics")
    dic_genre_word_count = unigram_model.read_files("Lyrics")

    # calculate tfidf scores for unigram
    uni_dic_tf = tf_idf_calculations.get_TF_values(dic_genre_word_count)
    uni_dic_idf = tf_idf_calculations.get_IDF_values(dic_genre_word_count)
    uni_dic_tfidf = tf_idf_calculations.get_TFIDF_values(uni_dic_tf, uni_dic_idf)

    # calculate tfidf scores for bigram
    bi_dic_tf = tf_idf_calculations.get_TF_values(dic_genre_pair_count)
    bi_dic_idf = tf_idf_calculations.get_IDF_values(dic_genre_pair_count)
    bi_dic_tfidf = tf_idf_calculations.get_TFIDF_values(bi_dic_tf, bi_dic_idf)

    # set lambdas
    best_lambda = mixed_model.set_lambda(uni_dic_tfidf, bi_dic_tfidf, "Validation Set")

    # evaluate models
    unigram_scores = []
    bigram_scores = []
    mixed_scores = []

    t1_uni_f1, t1_bi_f1, t1_mixed_f1 = evaluator1.evaluate_f1("C:\\Users\\Gigabyte\\Desktop\\Text Mining\\assignment2\\test.tsv", uni_dic_tfidf, bi_dic_tfidf, best_lambda)
    t2_uni_f1, t2_bi_f1, t2_mixed_f1 = evaluator1.evaluate_f1("C:\\Users\\Gigabyte\\Desktop\\Text Mining\\assignment2\\test2.tsv", uni_dic_tfidf, bi_dic_tfidf, best_lambda)

    unigram_scores.append(t1_uni_f1)
    unigram_scores.append(t2_uni_f1)
    bigram_scores.append(t1_bi_f1)
    bigram_scores.append(t2_bi_f1)
    mixed_scores.append(t1_mixed_f1)
    mixed_scores.append(t2_mixed_f1)

    # prints f1 scores
    print("Unigram f1 scores = ", unigram_scores[0], ",", unigram_scores[1], " Bigram f1 scores = ", bigram_scores[0], ",", bigram_scores[1], " Mixed f1 scores = ", mixed_scores[0], ",", mixed_scores[1])
    
    evaluator1.evaluate_significance(unigram_scores, bigram_scores, mixed_scores)

main()
