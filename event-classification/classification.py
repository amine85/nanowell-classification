from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid

class Classify:
    """ wrapper class around scikit-learn classification functions"""
  
    def __init__(self, data = None, target = None, 
                       n_class = None, clf_type = None,
                       params = None):
        self.data = data
        self.target = target
        self.n_class = n_class
        self.clf_type = clf_type
        self.params = params
        self._classifier_options = {"SVM":self._svm,
                                  "KNN":self._knn,
                                  "NBAYES":self._naive_bayes,
                                  "DTREES":self._decision_tree,
                                  "ADABOOST":self._adaboost}
                                  
    ##############################################################################
    # public interface
    ############################################################################## 
    def set_data(self, data = None):
        """ set input data (overwrites current data)"""
        self.data = data                                    

    def set_target(self, target = None):
        """ set targets (overwrites current targets)"""
        self.target = target            
        
    def set_num_classes(self, n_class = None):
        """ set number of classes (overwrites current number)"""
        self.n_class = n_class                    
        
    def set_classifier_type(self, clf_type = None):
        """ set classifier type (overwrites current type)"""
        self.clf_type = clf_type                            
        
    def get_data(self):
        """ return the data if it exists"""
        if self.data:
            return self.data        
        else:
            print "data does not exist"
            return None
    
    def get_target(self):
        """ return the target if it exists"""
        if self.target:
            return self.target        
        else:
            print "target does not exist"
            return None        

    def get_num_classes(self):
        """ return the number of classes if it exists"""
        if self.n_class:
            return self.n_class        
        else:
            print "number of classes does not exist"
            return None        

    def get_classifier_type(self):
        """ return the classfier type if it exists"""
        if self.clf_type:
            return self.clf_type        
        else:
            print "classifier type does not exist"
            return None        
    def classify(self):
        """ classify the data into n_class"""        
        if not self.clf_type:
            print "You must input a method type with 'set_classifier_type()',"
            print "choose from one of the following:"
            for key,_ in self._classifier_options.items():
                 print key 
            return None       
                
        if self._check_inputs():
            self._classifier_options[self.clf_type] 
            
    ##############################################################################
    # private interface
    ##############################################################################    
    def _check_inputs(self):
        """ verifies if all inputs are there"""
        if not self.data:
            print "missing input data"
            return False
        if not self.target:
            print "missing input labels"
            return False
        if not self.n_class:
            print "missing number of classes"
            return False
        if not self.params:
            print "missing parameters"
            return False
        return True
    
    def _check_svm_params(self):
        """ check if svm parameters are there"""        

    def _svm(self):
        """ svm classifier"""
        
        
        
        clf = svm.SVC()
        clf.fit(self.data, self.target, kernel='rbf') 
        self.predicted = clf.predict(self.target)
        return None
        
    def __knn(self):
        """ knn classifier"""
        clf = NearestCentroid()
        clf.fit(self.dataTraining, self.dataLabel,)
        NearestCentroid(metric='euclidean', shrink_threshold=None)
        return None
        

        







