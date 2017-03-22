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
    
def is_continuous(colname, dmatrix):
    '''
    Check if the colname was treated as continuous in the patsy.dmatrix
    Would look like colname[<factor_value>] otherwise
    '''
    return colname in dmatrix.columns

def ks_boot(tr, co, nboots=1000):
    nx = len(tr)
    ny = len(co)
    w = tr.append(co)
    obs = len(w)
    cutp = nx
    ks_boot_pval = None
    bbcount = 0
    ss = []
    fs_ks, _ = stats.ks_2samp(tr, co)
    for bb in range(nboots):
        sw = np.random.choice(w, obs, replace=True)
        x1tmp = sw[:cutp]
        x2tmp = sw[cutp:]
        s_ks, _ = stats.ks_2samp(x1tmp, x2tmp)
        ss.append(s_ks)
        if s_ks >= fs_ks:
            bbcount += 1
    ks_boot_pval = bbcount / nboots
    return ks_boot_pval
    
class Matcher:

    def __init__(self, test, control, formula=None, 
                 yvar='treatment', exclude=[], stepwise=False, transform=False):
        self.yvar = yvar   
        self.exclude = exclude + [self.yvar] + ['scores', 'match_id']
        self.formula = formula
        self.models = []
        self.swdata = None
        self.model_accurracy = []
        self.stepwise = stepwise
        self.transform = transform
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
        if self.stepwise or self.transform:
            for col in self.xvars:
                if is_continuous(col, self.X):
                    l = 'log_' + col
                    s = 'sqrt_' + col
                    p = 'pow2_' + col
                    self.xvars.extend([l, s, p])
                    self.data[l] = np.log(self.data[col] + 1)
                    self.data[s] = np.sqrt(self.data[col] + 1)
                    self.data[p] = np.power(self.data[col], 2)     
            self.y, self.X = patsy.dmatrices('{} ~ {}'.format(yvar, '+'.join(self.xvars)), data=self.data,
                                             return_type='dataframe')

        self.xvars = [i for i in self.data.columns if i not in exclude]
            
        self.test= self.data[self.data[yvar] == True]
        self.control = self.data[self.data[yvar] == False]
        self.testn = len(self.test)
        self.controln = len(self.control)
        
        # do we have a class imbalance problem?
        self.minority, self.majority = \
          [i[1] for i in sorted(zip([self.testn, self.controln], [1, 0]), 
                                key=lambda x: x[0])]
            
        print 'n majority:', len(self.data[self.data[yvar] == self.majority])
        print 'n minority:', len(self.data[self.data[yvar] == self.minority])
        
        # explodes design matrix if included
        assert "client_id" not in self.xvars, \
               "client_id shouldn't be a covariate! Please set exclude=['client_id']"    

    def fit_scores(self, balance=True, nmodels=None, k=3):       
        if not self.formula:
            # use all columns in the model (untransformed)
            self.formula = '{} ~ {}'.format(self.yvar, '+'.join(self.xvars))
            if self.stepwise:
                print "Optimizing Forumla via forward stepwise selection..."
                # use all columns + trasnformed columns in model
                self.formula, self.swdata = \
                   self.forward_stepwise(self.balanced_sample(), self.yvar, k=k)
        if balance:
            if nmodels is None:
                # fit mutliple models based on imbalance severity (rounded up to nearest tenth)
                minor, major = [self.data[self.data[self.yvar] == i] for i in (self.minority, self.majority)]
                nmodels = int(np.ceil((len(major) / len(minor)) / 10) * 10)
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
            print '\nFitting 1 (Unbalanced) Model...'
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
            scores += m.predict(self.X[m.params.index])
        self.data['scores'] = scores/self.nmodels
        
    def match(self, threshold=0.001, nmatches=1, tie_strategy='random', max_rand=10):
        if 'scores' not in self.data.columns:
            print "Propensity Scores have not been calculated. Using defaults..."
            self.fit_scores()
            self.predict_scores()
        test_scores = self.data[self.data[self.yvar]==True][['scores']]
        ctrl_scores = self.data[self.data[self.yvar]==False][['scores']]
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
        
    # adapted from 
    # http://planspace.org/20150423-forward_selection_with_statsmodels/
    def forward_stepwise(self, data, resp, k=5):
        selected = []
        remaining = set(data.columns)
        for i in self.exclude:
            try:
                remaining.remove(i)
            except KeyError:
                continue

        current_score, best_new_score = 0.0, 0.0
        ret_data = []
        regressions = 0
        i = 1
        while remaining and regressions < 5:
            scores_with_candidates = []
            for candidate in remaining:
                progress(current_score, 1, "n_models={}, Current Best: ".format(i))
                i += 1
                formula = "{} ~ {}".format(resp,
                                               ' + '.join(selected + [candidate]))
                data_part = data[[resp] + selected + [candidate]]

                y = data_part[self.yvar]
                X = self.select_from_design(data_part.columns)

                train, test = self.kfold_cv(data_part, formula, k=k)

                scores_with_candidates.append((test, candidate))
                ret_data.append({'f':formula, 'train':train, 'test':test})
            scores_with_candidates.sort()
            best_new_score, best_candidate = scores_with_candidates.pop()
            if best_new_score - current_score > .001:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                current_score = best_new_score
                regressions = 0
            else:
                regressions += 1

        formula = "{} ~ {}".format(resp,
                                       ' + '.join(selected))
        return formula, ret_data

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

            y = train[[self.yvar]]
            X = self.select_from_design(train.columns).loc[train.index]
            yt = test[[self.yvar]]
            Xt = self.select_from_design(test.columns).loc[test.index]
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
        
        
    def select_from_design(self, cols):
        d = pd.DataFrame()
        for c in cols:
            d = pd.concat([d, self.X.select(lambda x: x.startswith(c), axis=1)], axis=1)
        return d

    def balanced_sample(self, data=None):
        if not data:
            data=self.data
        minor, major = data[data[self.yvar] == self.minority], data[data[self.yvar] == self.majority]
        return major.sample(len(minor)).append(minor).dropna()

    def plot_scores(self):
        assert 'scores' in self.data.columns, "Propensity scores haven't been calculated, use Matcher.predict_scores()"
        sns.distplot(self.data[self.data[self.yvar]==True].scores, label='Test')
        sns.distplot(self.data[self.data[self.yvar]==False].scores, label='Control')
        plt.legend(loc='upper right')
        
        
    def ks_by_column(self):
        def split_and_test(data, column):
            ctest = data[data[self.yvar] == True][column]
            cctrl = data[data[self.yvar] == False][column]
            return ks_boot(ctest, cctrl, nboots=500)
        
        data = []
        #assert len(self.matched_data) > 0, 'Data has not been matched, use Matcher.match()'
        for column in self.data.columns:
                if column not in self.exclude and is_continuous(column, self.X):
                    pval_before = split_and_test(self.data, column)
                    pval_after = split_and_test(self.matched_data, column)
                    
                    data.append({'var': column, 
                                 'p_before': round(pval_before, 6), 
                                 'p_after': round(pval_after, 6)})
        return pd.DataFrame(data)[['var', 'p_before', 'p_after']]
    
    def prop_test_by_column(self):
        def prep_prop_test(self, data, var):
            t, c = data[data[self.yvar]==True], data[data[self.yvar]==False]
            countt = t[[var, 'client_id']].groupby(var).count() / len(t)
            countc = c[[var, 'client_id']].groupby(var).count() / len(c)

            ignore = list(set(countt.index) ^ set(countc.index))

            for t in (countt, countc):
                try:
                    t.client_id.drop(ignore, inplace=True)
                except ValueError:
                    # not in axis
                    pass
            return [list(countt.client_id), list(countc.client_id)]

        ret = []
        for col in self.matched_data.columns:
            if not is_continuous(col, self.X) and col not in self.exclude and 'final' not in col:
                pval_before = round(stats.chi2_contingency(prep_prop_test(self, self.data, col))[1], 6)
                pval_after = round(stats.chi2_contingency(prep_prop_test(self, self.matched_data, col))[1], 6)
                ret.append({'var':col, 'before':pval_before, 'after':pval_after})
        return pd.DataFrame(ret)[['var', 'before', 'after']]


            

                   
                    
    def plot_ecdfs(self):
        for col in self.matched_data.columns:
            if is_continuous(col, self.X) and col not in self.exclude:
                xtb, xcb = ECDF(self.test[col]), ECDF(self.control[col])
                xta, xca = ECDF(self.matched_data[self.matched_data[self.yvar]==True][col]),\
                   ECDF(self.matched_data[self.matched_data[self.yvar]==False][col])

                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 5))

                ax1.plot(xcb.x, xcb.y, label='Control')
                ax1.plot(xtb.x, xtb.y, label='Test')
                ax1.set_title('ECDF for {} before Matching'.format(col))

                ax2.plot(xca.x, xca.y, label='Control')
                ax2.plot(xta.x, xta.y, label='Test')
                ax2.set_title('ECDF for {} after Matching'.format(col))
                ax2.legend(bbox_to_anchor=(1.4, 1.03))

                plt.xlim((0, np.percentile(xta.x, 99)))
                
    def plot_bars(self):
        def prep_plot(data, var, colname):
            t, c = data[data[self.yvar]==True], data[data[self.yvar]==False]
            countt = t[[var, 'client_id']].groupby(var).count() / len(t)
            countc = c[[var, 'client_id']].groupby(var).count() / len(c)
            ret = (countt-countc).dropna()
            ret.columns = [colname]
            return ret
        
        for col in self.matched_data.columns:
            if not is_continuous(col, self.X) and col not in self.exclude and 'final' not in col:
                dbefore = prep_plot(self.data, col, colname='before')
                dafter = prep_plot(self.matched_data, col, colname='after')

                dbefore.join(dafter).plot.bar()
                plt.title('Proportional Difference (test - control)')
            
        
    def _scores_to_accuracy(self, m, X, y):
        preds = [1.0 if i >= .5 else 0.0 for i in m.predict(X)]
        return (y == preds).sum() / len(y)