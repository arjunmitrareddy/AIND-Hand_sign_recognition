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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        """ select the best model for self.this_word based on
                BIC score for n between self.min_n_components and self.max_n_components

                :return: GaussianHMM object
                """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_model = None
        best_score = float("inf")
        for num_states in range(self.min_n_components, self.max_n_components):
            try:
                hmmmodel = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmmmodel.score(self.X, self.lengths)
                initialStateProbs = num_states
                transitionProbs = num_states * (num_states - 1)
                emissionProbs = len(np.diagonal(hmmmodel.means_)) + len(np.diagonal(hmmmodel.covars_))
                p = initialStateProbs + transitionProbs + emissionProbs
                BIC_Score = -2 * logL + p * np.log(len(self.X))
                if BIC_Score < best_score:
                    best_score = BIC_Score
                    best_model = hmmmodel
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
        best_model = None
        best_score = float("-inf")
        for num_states in range(self.min_n_components, self.max_n_components):
            try:
                hmmmodel = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                       random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL_own = hmmmodel.score(self.X, self.lengths)

                logL_other_words = []
                for word in self.hwords:
                    if word == self.this_word:
                        continue
                    other_X, other_L = self.hwords[word]
                    logL_other_words.append(hmmmodel.score(other_X, other_L))

                average_other_logL = np.average(logL_other_words)
                DIC_score = logL_own - average_other_logL
                if DIC_score > best_score:
                    best_score = DIC_score
                    best_model = hmmmodel
            except:
                continue
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float("-inf")
        for num_states in range(self.min_n_components, self.max_n_components):
            if len(self.sequences) == 1 :
                continue
            split_method = KFold(n_splits = len(self.sequences) if len(self.sequences) < 3 else 3)
            iter_scores = []
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                try:
                    X, L = combine_sequences(cv_train_idx, self.sequences)
                    hmmmodel = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state, verbose=False).fit(X, L)
                    X_test, L_test = combine_sequences(cv_test_idx, self.sequences)
                    logL = hmmmodel.score(X_test, L_test)
                    iter_scores.append(logL)
                except:
                    continue
                avg_iter_score = np.average(iter_scores)
                if avg_iter_score > best_score:
                    best_model = hmmmodel
                    best_score = avg_iter_score
        return best_model



