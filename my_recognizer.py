import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    probabilities = []
    guesses = []
    x_lengths = test_set.get_all_Xlengths()

    for sequence in test_set.get_all_sequences():
        x, lengths = x_lengths[sequence]
        best_guess = ""
        max_score = float("-inf")
        prob_dist = {}

        for word, model in models.items():
            try:
                log_loss = model.score(x, lengths)
            except:
                log_loss = float("-inf")
            
            prob_dist[word] = log_loss
            if log_loss > max_score:
                best_guess = word
                max_score = log_loss
        
        probabilities.append(prob_dist)
        guesses.append(best_guess)
    
    return probabilities, guesses
