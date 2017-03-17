from __future__ import division
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
import statsmodels.api as sm
import patsy 
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def progress(i, n, prestr=''):
    sys.stdout.write('\r{}{}%'.format(prestr, round(i / n * 100, 2)))
    
def is_continuous(colname, dmatrix, cutoff=.05):
    '''
    Check if the colname was treated as continuous in the patsy.dmatrix
    
    Would look like colname[<factor_value>] otherwise
    '''
    return colname in dmatrix.columns
    
class Matcher:
    '''
    Matcher Class
    '''
    
    def __init__(self, test, control, formula=None, 
                 yvar='treatment', exclude=[], stepwise=False):
        self.yvar = yvar   
        self.exclude = exclude + [self.yvar] + ['scores']
        self.formula = formula
        self.models = []
        self.model_accurracy = []
        self.stepwise = stepwise
        # create unique indices for each row
        # and combine test and control
        t, c = [i.copy().reset_index(drop=True) for i in (test, control)]
        c.index += len(t)
        self.data = t.append(c).dropna()
        # should be binary 0, 1
        self.data[yvar] = self.data[yvar].astype(int)
        self.xvars = [i for i in self.data.columns if i not in self.exclude]
        self.matched_data = []  
        # create design matrix of all variables not in <exclude>
        self.y, self.X = patsy.dmatrices('{} ~ {}'.format(yvar, '+'.join(self.xvars)), data=self.data,
                                         return_type='dataframe')
        
        # add some transformations for stepwise procress
        if self.stepwise:
            for col in self.xvars:
                if is_continuous(col, self.X):
                    self.data['log_' + col] = np.log(self.data[col] + 1)
                    self.data['sqrt_' + col] = np.sqrt(self.data[col] + 1)
                    self.data['pow2_' + col] = np.power(self.data[col], 2)     

            self.xvars = [i for i in self.data.columns if i not in exclude]
            
        self.test= self.data[self.data[yvar] == True]
        self.control = self.data[self.data[yvar] == False]
        self.testn = len(self.test)
        self.controln = len(self.control)
        
        # explodes design matrix if included
        assert "client_id" not in self.xvars, \
               "client_id shouldn't be a covariate! Please set exclude=['client_id']"    

    def fit_scores(self, balance=True, nmodels=None):
        # do we have a class imbalance problem?
        self.minority, self.majority = \
          [i[1] for i in sorted(zip([self.testn, self.controln], [self.test, self.control]), 
                                key=lambda x: x[0])]
        print "n_majority: {}\nn_minority: {}".format(len(self.majority), len(self.minority))
        
        if not self.formula:
            # use all columns in the model (untransformed)
            self.formula = '{} ~ {}'.format(self.yvar, '+'.join(self.xvars))
            if self.stepwise:
                # use all columns + trasnformed columns in model
                print "Optimizing Forumla via forward stepwise selection..."
                self.formula = self.forward_stepwise(self.data, self.yvar, k=5)
                self.y, self.X = patsy.dmatrices(self.formula, data=self.data, 
                                return_type='dataframe')
        if balance:
            if nmodels is None:
                # fit mutliple models based on imbalance severity (rounded up to nearest tenth)
                nmodels = int(np.ceil((len(self.majority) / len(self.minority)) / 10) * 10)
            self.nmodels = nmodels
            for i in range(nmodels):
                progress(i+1, nmodels, 
                         prestr="Fitting {} Models on Balanced Samples...".format(nmodels))
                
                # sample from majority to create balance dataset
                df = self.balanced_sample()  
                y_samp, X_samp = patsy.dmatrices(self.formula, data=df, return_type='dataframe')
                glm = GLM(y_samp, X_samp, family=sm.families.Binomial())
                res = glm.fit()
                self.model_accurracy.append(self._scores_to_accuracy(res, X_samp, y_samp))
                self.models.append(res)
            print "\nAverage Accuracy:", "{}%".\
                  format(round(np.mean(self.model_accurracy) * 100, 2))
        else:
            # ignore any imbalance and fit one model
            self.nmodels = 1
            print 'Fitting 1 (Unbalanced) Model...'
            glm = GLM(self.y, self.X, family=sm.families.Binomial())
            res = glm.fit()
            self.model_accurracy.append(self._scores_to_accuracy(res, self.X, self.y))
            self.models.append(res)
            print "Accuracy", round(np.mean(self.model_accurracy[0]) * 100, 2)
            
    def predict_scores(self):
        scores = np.zeros(len(self.X))
        for i in range(self.nmodels):
            progress(i+1, self.nmodels, "Caclculating Propensity Scores...")
            m = self.models[i]
            X_dup = self.X.copy()
            # ignore columns not found in trained model
            for col in X_dup.columns:
                if col not in m.params.index:
                    X_dup.drop(col, axis=1, inplace=True)
            scores += m.predict(X_dup)
        self.data['scores'] = scores/self.nmodels
        
    def match(self, threshold=0.001, nmatches=1, tie_strategy='random', max_rand=10):
        if 'scores' not in self.data.columns:
            print "Propensity Scores have not been calculated. Using defaults..."
            self.fit_scores()
            self.predict_scores()
        test_scores = self.data[self.data.treatment==True][['scores']]
        ctrl_scores = self.data[self.data.treatment==False][['scores']]
        result, match_ids = [], []
        for i in range(len(test_scores)):
            progress(i+1, len(test_scores), 'Matching Control to Test...')
            match_id = i
            score = test_scores.iloc[i]
            if tie_strategy == 'random':
                bool_match = abs(ctrl_scores - score) <= threshold
                matches = ctrl_scores.loc[bool_match[bool_match.scores].index]
            elif tie_strategy == 'min':
                matches = abs(ctrl_scores - score).sort_values('scores').head(1)
            else:
                raise AssertionError, "Invalid tie_strategy parameter, use ('random', 'min')"
            if len(matches) == 0:
                continue
            # randomly choose nmatches indices, if len(matches) > nmatches
            select = nmatches if nmatches != 'random' else np.random.choice(range(1, max_rand+1), 1)
            chosen = np.random.choice(matches.index, min(select, len(matches)), replace=False)
            result.extend([test_scores.index[i]] + list(chosen))
            match_ids.extend([i] * (len(chosen)+1))
        self.matched_data = self.data.loc[result]
        self.matched_data['match_id'] = match_ids  
        
    def forward_stepwise(self, data, resp, k=5):
        selected = []
        remaining = set(data.columns)
        for i in self.exclude:
            try:
                remaining.remove(i)
            except KeyError:
                continue

        current_score, best_new_score = 0.0, 0.0
        while remaining and current_score == best_new_score:
            scores_with_candidates = []
            best_score = best_new_score
            for candidate in remaining:
                formula = "{} ~ {}".format(resp,
                                               ' + '.join(selected + [candidate]))
                data_part = data[[resp] + selected + [candidate]]

                y, X = patsy.dmatrices(formula, data=data_part, 
                            return_type='dataframe')
                train, test = self.kfold_cv(data_part, formula, k=k)

                scores_with_candidates.append((test, candidate))
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            if current_score < best_new_score and best_new_score > 0:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
        formula = "{} ~ {}".format(resp,
                                       ' + '.join(selected))
        return formula

    def kfold_cv(self, d, formula, k):
        n = len(d)
        d = d.sample(n, replace=False)
        partition = n // k
        current, last = 0, partition
        train_accs = []
        test_accs = []
        while current < n:
            if last > n - partition:
                last = n
            test = d.iloc[current:last]
            train = d.drop(test.index)

            y, X = patsy.dmatrices(formula, data=train, 
                            return_type='dataframe')
            yt, Xt = patsy.dmatrices(formula, data=test, 
                            return_type='dataframe')

            shared = list(set(Xt.columns) & set(X.columns))
            glm = GLM(y, X[shared], family=sm.families.Binomial())
            try:
                res = glm.fit()
                train_acc = self._scores_to_accuracy(res, X[shared], y)
                test_acc = self._scores_to_accuracy(res, Xt[shared], yt)
                train_accs.append(train_acc)
                test_accs.append(test_acc)
            except PerfectSeparationError:
                print "Perfectly Separated!"
            current = last
            last += partition
        return np.mean(train_accs), np.mean(test_accs)
        

    def balanced_sample(self):
        return self.majority.sample(len(self.minority)).append(self.minority).dropna()

    def plot_scores(self):
        assert 'scores' in self.data.columns, "Propensity scores haven't been calculated, use Matcher.predict_scores()"
        sns.distplot(self.data[self.data[self.yvar]==True].scores, label='Test')
        sns.distplot(self.data[self.data[self.yvar]==False].scores, label='Control')
        plt.legend(loc='upper right')
        
        
    def ks_by_column(self):
        def split_and_test(data, column):
            ctest = data[data.treatment == True][column]
            cctrl = data[data.treatment == False][column]
            return stats.ks_2samp(ctest, cctrl)
        
        data = []
        #assert len(self.matched_data) > 0, 'Data has not been matched, use Matcher.match()'
        for column in self.data.columns:
                if column not in self.exclude and is_continuous(column, self.X):
                    _, pval_before = split_and_test(self.data, column)
                    _, pval_after = split_and_test(self.matched_data, column)
                    
                    data.append({'var': column, 
                                 'p_before': round(pval_before, 6), 
                                 'p_after': round(pval_after, 6)})
        return pd.DataFrame(data)[['var', 'p_before', 'p_after']]
                   
                    
    def plot_ecdfs(self):
        for col in self.matched_data.columns:
            if is_continuous(col, self.X):
                xtb, xcb = ECDF(self.test[col]), ECDF(self.control[col])
                xta, xca = ECDF(self.matched_data[self.matched_data.treatment==True][col]),\
                   ECDF(self.matched_data[self.matched_data.treatment==False][col])

                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))

                ax1.plot(xcb.x, xcb.y, label='Control')
                ax1.plot(xtb.x, xtb.y, label='Test')
                ax1.set_title('ECDF for {} before Matching'.format(col))

                ax2.plot(xca.x, xca.y, label='Control')
                ax2.plot(xta.x, xta.y, label='Test')
                ax2.set_title('ECDF for {} after Matching'.format(col))
                ax2.legend(bbox_to_anchor=(1.4, 1.03))

                plt.xlim((0, np.percentile(xta.x, 99)))
            
        
    def _scores_to_accuracy(self, m, X, y):
        preds = [1.0 if i >= .5 else 0.0 for i in m.predict(X)]
        return (y == preds).sum() / len(y)