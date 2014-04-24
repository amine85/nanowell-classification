from sklearn.decomposition import PCA,KernelPCA,NMF,RandomizedPCA,TruncatedSVD
from sklearn.lda import LDA


class DimensionReduction:
    """ wrapper class around scikit-learn dimensionality reduction functions"""

    def __init__(self, data = None, target = None, method = None, params = None):
        self.data = data
        self.target = target
        self.method = method
        self.params = params                    # this is dictionary of dictionaries of parameters
        self._tranform_options = {"PCA":self._pca,
                                  "K-PCA":self._kpca,
                                  "ApproxPCA":self._rand_pca,
                                  "LDA":self._lda,
                                  "NMF":self._nmf,
                                  "TSVD":self._truncated_svd}
    ##############################################################################
    # public interface
    ##############################################################################                                   
    def set_data(self,data = None, target = None):
        """ set the data"""
        self.data = data
        self.target = target

    def set_method(self, method = None):
        """ set the method"""
        self.method = method

    def set_params(self, params = None):
        """ set the method"""
        self.params = params
        
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
    
    def get_method(self):
        if self.method:
            return self.method
        else:
            print "no method was defined previously"
            return None
                       
    def transform(self, method = None):
        """ reduces the dimension of the data"""        
        if not self.method:
            if method:
                self.method = method
            else:
                print "You must input a method type,"
                print "choose from one of the following:"
                for key,_ in self.method_options.items():
                    print key 
                return None
        if not (self.params and self.data):
            print "either data or parameters or both have not been set"
            return
        if not ("n_components" in self.params[self.method].keys()):
            print "the number of compoents has not been set"
            return
        self.n = self.params[self.method]["n_components"]    
        self._tranform_options[self.method]   
        
    ##############################################################################
    # private interface
    ##############################################################################    
    def _pca(self):
        """ performs princpal components analysis of the data"""
        pca = PCA(n_components=self.n)
        self.transformed_data = pca.fit_transform(self.data).transform(self.data)    
        
    def _lda(self):                              
        """ performs linear discriminant analysis of the data"""
        if not self.target:
            print "target has not been set, it is required for LDA"
            return 
        lda = LDA(n_components=self.n)
        self.transformed_data = lda.fit_transform(self.data, self.target).transform(self.data) 

    def _kpca(self):
        """ performs transform using kernel principal component analysis,
            this function uses;
            kernel: radial basis function 
            eigen_solver: arpack                           """    
        if not ("gamma" in self.params[self.method].keys()):
            print "need to set the rbf kernel parameter"
            return
           
        g = self.params[self.method]["gamma"]
        kpca = KernelPCA(n_components=self.n, kernel="rbf", gamma = g, eigen_solver = "arpack")
        self.transformed_data = kpca.fit_transform(self.data)       
        
    def _rand_pca(self):
        """ performs apprixamte principal components analysis,
            this useful for data sets with a large number of features"""        
        rpca = RandomizedPCA(n_components=self.n,whiten= True )
        self.transformed_data = rpca.fit_transform(self.data)    
        
    def _nmf(self):
        """ performs non-negative matrix factorization
           this method does not enforce sparseness on the coefficients """
        nmf = NMF(n_components=self.n, tol=5e-3)
        self.transformed_data = nmf.fit_transform(self.data)
    def _truncated_svd(self):
        """ performs truncated singular value decomposition 
            this function uses:
            algorithm solver: arpack                           """    
        svd = TruncatedSVD(n_component = self.n, algorithm = "arpack")
        self.transformed_data = svd.fit_transform(self.data)

     
    
        
        








