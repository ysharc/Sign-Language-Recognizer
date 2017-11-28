import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        min_score = float("inf")
        best_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                log_likelihood = model.score(self.X, self.lengths)
                n_features = self.X.shape[1] # no. of features
                n_parameters = num_states**2 + 2 * n_features * num_states - 1
                N = self.X.shape[0]
                bic = -2 * log_likelihood + n_parameters * np.log(N)
                if bic < min_score:
                    min_score = bic
                    best_model = model
            except:
                continue

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_score = float("-inf")
        best_model = None

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                word_likelihood = model.score(self.X, self.lengths)
                tot_anti_likelihood = 0
                
                for word in self.words:
                    if word != self.this_word:
                        diff_x, diff_lengths = self.hwords[word]
                        # The average anti likelihood is added finally to the likelihood
                        # This is the reason why the negative model score is added below.
                        tot_anti_likelihood += -model.score(diff_x, diff_lengths)
                avg_anti_likelihood = tot_anti_likelihood / (len(self.words) - 1)
                dic = word_likelihood + avg_anti_likelihood

                if dic > max_score:
                    best_model = model
                    max_score = dic
            except:
                pass

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        max_score = float("-inf")
        best_model = None
        n_splits = 3

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            if len(self.sequences) < n_splits:
                break

            split_method = KFold(n_splits=n_splits)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_data, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                test_data, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                try:
                    model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                random_state=self.random_state, verbose=False).fit(train_data, train_lengths)
                    log_likelihood = model.score(test_data, test_lengths)
                    scores.append(log_likelihood)
                except ValueError:
                    break

            if scores:
                avg_log_likelihood = np.mean(scores)
            else:
                avg_log_likelihood = float("-inf")

            if avg_log_likelihood > max_score:
                max_score = avg_log_likelihood
                best_model = model
        
        if not best_model:
            return self.base_model(self.n_constant)
        return best_model
