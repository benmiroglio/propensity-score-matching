from Matcher import *

def get_sample_data():
    import urllib2
    import re
    path='https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/iris.csv'
    iris = pd.read_csv(urllib2.urlopen(path), index_col=0)
    # make binary
    iris['treatment'] = [i in ('setosa', 'virginica') for i in iris.Species]
    iris = iris.drop('Species', axis=1)
    iris.columns = [re.sub('\.', '', i) for i in iris.columns]
    test, control = iris[iris.treatment == True], iris[iris.treatment == False]
    return test, control

THRESHOLD = 0.01

test, control = get_sample_data()
m = Matcher(test, control)
m.match(threshold=THRESHOLD)
    
def test_indices():
    # control index should be shifted by test index
    assert max(m.test.index) + 1 == min(m.control.index)
    
def test_match_properties():
    for match_id in np.unique(m.matched_data.match_id):
        current = m.matched_data[m.matched_data.match_id == match_id]
        s1, s2 = current.scores
        t1, t2 = current.treatment
        
        # matched scores should be within THRESHOLD
        assert abs(s1 - s2) <= THRESHOLD
        
        # should be [True, False] or [False, True]
        # when matching 1 control profile to 1 test profile
        assert sum([t1, t2]) == 1
        
        
def test_no_match_drop():
    # as the threshold decreases, so should the size of matched data
    last_n = float("inf")
    for t in [0.1, 0.01, 0.001]:
        m.match(threshold=t)
        n = len(m.matched_data)
        assert n <= last_n
        last_n = n
        
    
        
    
    