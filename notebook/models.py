from __future__ import division
import numpy as np
import scipy.sparse as sparse
import copy
from scipy.stats import binom, hypergeom, gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import chi2
from sklearn.preprocessing import scale

def scale_X(X):
    X = X.astype(float)
    if issparse(X):
        X = scale(X, with_mean=False)
    else:
        X = scale(X)
    return X

# Basic Models
def lr(X, y, c, rand, feature_names):
    clf = LogisticRegression(class_weight='')
    #X = scale_X(X)
    clf.fit(X, y)
    return clf

def lin_svc(X, y, c, rand, feature_names):
    clf = LinearSVC()
    clf.fit(X,y)
    return clf

def nb(X, y, c, rand, feature_names):
    clf = BernoulliNB()
    #X = scale_X(X)
    clf.fit(X, y)
    return clf

# A matching-based classifier.
def make_same_length(a, b):
    # Duplicate the smaller list until it is at least as large as the larger list.
    if len(a) < len(b):
        factor = int(math.ceil(1. * len(b) / len(a)))
        a = a * factor
    else:
        factor = int(math.ceil(1. * len(a) / len(b)))
        b = b * factor
    return a, b        

    
def matching_sum(X, y, c, rand, feature_names):
    """
    For each training example where y=y_i and c=c_i, create a negative example equal to the mean
    feature value for y=y_i' and c=c_i.
    Training objective is to discriminate these pairs of examples.
    FIXME: this is slow.
    """
    yc_eq = set(np.where(y == c)[0])
    yc_diff = set(np.where(y != c)[0])
    ypos = set(np.where(y == 1)[0])
    yneg = set(np.where(y == 0)[0])
    
    both_pos = list(yc_eq & ypos)
    both_neg = list(yc_eq & yneg)
    ypos_cneg =list(yc_diff & ypos)
    yneg_cpos =list(yc_diff & yneg)
    
    both_pos_sum = X[both_pos, :].mean(axis=1)
    both_neg_sum = X[both_neg, :].mean(axis=1)
    ypos_cneg_sum = X[ypos_cneg, :].mean(axis=1)
    yneg_cpos_sum = X[yneg_cpos, :].mean(axis=1)

    #X = scale_X(X)
    rows = []
    newY = []
    flip = 1.
    for i in range(len(y)):
        if y[i] == 1:
            if c[i] == 1:
                rows.append((X[i] - yneg_cpos_sum) * flip)
                newY.append(max(int(flip), 0))
                flip *= -1
                #for j in yneg_cpos:
                #    if j > i:
                #        rows.append((X[i] - X[j]) * flip)
                #        newY.append(max(int(flip), 0))
                #        flip *= -1
            else:
                rows.append((X[i] - both_neg_sum) * flip)
                newY.append(max(int(flip), 0))
                flip *= -1                
                #for j in both_neg:
                #    if j > i:
                #        rows.append((X[i] - X[j]) * flip)
                #        newY.append(max(int(flip), 0))
                #        flip *= -1
        else:
            if c[i] == 1:
                rows.append((ypos_cneg_sum - X[i]) * flip)
                newY.append(max(int(flip), 0))
                flip *= -1                
                #for j in ypos_cneg:
                #    if j > i:
                #        rows.append((X[j] - X[i]) * flip)
                #        newY.append(max(int(flip), 0))
                #        flip *= -1
            else:
                rows.append((both_pos_sum - X[i]) * flip)
                newY.append(max(int(flip), 0))
                flip *= -1                                
                #for j in both_pos:
                #    if j > i:
                #        rows.append((X[j] - X[i]) * flip)
                #        newY.append(max(int(flip), 0))
                #        flip *= -1
                    
    newX = sparse.vstack(rows)
    m = LogisticRegression(fit_intercept=False, class_weight="")
    print('fit on %d instances' % newX.shape[0])
    m.fit(newX, newY)
    return m                

# problem when dataset does not contain instances for all y-c pairs.
def matching(X, y, c, rand, feature_names):
    """
    For each training example where y=y_i and c=c_i, create a negative example by sampling 
    an instance where y!=y_i and c=c_i
    Training objective is to discriminate these pairs of examples.
    FIXME: accuracy is pretty low. Is it because p(c) is biases in the sampling method?
    """
    yc_eq = set(np.where(y == c)[0])
    yc_diff = set(np.where(y != c)[0])
    ypos = set(np.where(y == 1)[0])
    yneg = set(np.where(y == 0)[0])
    
    both_pos = list(yc_eq & ypos)
    both_neg = list(yc_eq & yneg)
    ypos_cneg =list(yc_diff & ypos)
    yneg_cpos =list(yc_diff & yneg)
    #print(ypos_cneg)

    rows = []
    newY = []
    flip = 1.
    for i in range(len(y)):
        if y[i] == 1:
            if c[i] == 1:
                fv = X[i] - X[rand.choice(yneg_cpos)]
            else:
                fv = X[i] - X[rand.choice(both_neg)]
        else:
            if c[i] == 1:
                fv = X[rand.choice(ypos_cneg)] - X[i]
            else:
                fv = X[rand.choice(both_pos)] - X[i]
            
        rows.append(fv * flip)
        newY.append(max(int(flip), 0))
        flip *= -1
                    
    newX = sparse.vstack(rows)
    m = LogisticRegression(fit_intercept=False, class_weight="")
    m.fit(newX, newY)
    return m    

# Sum out the confounding variable.
# 1. Fit classifier in product space of c/y. (E.g., P(c^y|x))
# 2. To classify new x, return P(c=1^y=1|x) + P(c=0^y=1|x)
class SumOutClf:
    def __init__(self, clf):
        self.clf = clf
        self.coef_ = [self.clf.coef_[2]]
        
    def predict(self, X):
        proba = np.matrix(self.clf.predict_proba(X))
        # sum cols 1,2 (y0c0, y0c1) and cols 3,4 (y1c0,y1c1)
        proba = np.hstack((proba[:,:2].sum(axis=1), proba[:,2:4].sum(axis=1)))
        # return argmax of each row
        return np.array(proba.argmax(axis=1).T.tolist()[0])
        
# problem on sumout when not all pairs y/c are present in the data
def sumout(X, y, c, rand, feature_names):
    clf = LogisticRegression(class_weight="")
    label2index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    yc = [label2index[(yi, ci)] for yi, ci in zip(y, c)]
    clf.fit(X, yc)
    return SumOutClf(clf)

def get_n_top_features(X, c, n, feature_names):
    chi, F = chi2(X, c)
    clf = LogisticRegression(class_weight="")
    clf.fit(X, c)
    coef_sign = clf.coef_[0] / np.abs(clf.coef_[0])
    signed_chi = chi * coef_sign
    counts = X.sum(0).tolist()[0]
    top_feats = [i for i in np.argsort(signed_chi)[::-1] if counts[i] > 1][:n]
    if feature_names is not None:
        print('top_feats=', feature_names[top_feats])
    return top_feats

def feature_select(X, y, c, rand, feature_names):
    """ Find the highest chi2 feature for class c and remove it from the classifier."""
    #X = scale_X(X)
    chi, F = chi2(X, c)
    clf = LogisticRegression(class_weight="")
    clf.fit(X, c)
    coef_sign = clf.coef_[0] / np.abs(clf.coef_[0])
    signed_chi = chi * coef_sign
    counts = X.sum(0).tolist()[0]
    top_feats = [i for i in np.argsort(signed_chi)[::-1] if counts[i] > 1][:1]
    if feature_names is not None:
        print('top_feats=', feature_names[top_feats])
    X2 = copy.copy(X)
    X2[:,top_feats] = 0.  # Set top feature to 0
    clf.fit(X2, y)
    return clf

def lr_subsampling(X, y, c, rand, feature_names):
    """
    Subsampling LR for binary label and binary confounder.
    """
    #X = scale_X(X)
    yc_eq = set(np.where(y == c)[0])
    yc_diff = set(np.where(y != c)[0])
    ypos = set(np.where(y == 1)[0])
    yneg = set(np.where(y == 0)[0])
    
    both_pos = list(yc_eq & ypos)
    both_neg = list(yc_eq & yneg)
    ypos_cneg =list(yc_diff & ypos)
    yneg_cpos =list(yc_diff & yneg)

    all_classes = [both_pos, both_neg, ypos_cneg, yneg_cpos]
    min_class = min([len(l) for l in all_classes if l])

    subsampled_idx = []
    for l in all_classes:
        if l:
            subsampled_class = np.random.choice(l, min_class, replace=False)
            subsampled_idx.extend(subsampled_class)

    yc_eq = set(np.where(y[subsampled_idx] == c[subsampled_idx])[0])
    yc_diff = set(np.where(y[subsampled_idx] != c[subsampled_idx])[0])
    ypos = set(np.where(y[subsampled_idx] == 1)[0])
    yneg = set(np.where(y[subsampled_idx] == 0)[0])
    
    both_pos = list(yc_eq & ypos)
    both_neg = list(yc_eq & yneg)
    ypos_cneg =list(yc_diff & ypos)
    yneg_cpos =list(yc_diff & yneg)

    all_classes = [both_pos, both_neg, ypos_cneg, yneg_cpos]
    #print([len(x) for x in all_classes])
    
    return lr(X[subsampled_idx],
              y[subsampled_idx],
              c[subsampled_idx], rand, feature_names)

class BackdoorAdjustment:      
    def __init__(self):
        self.clf = LogisticRegression(class_weight='')
    
    def predict_proba(self, X):                                               
        # build features with every possible confounder                       
        l = X.shape[0]                                                        
        rows = range(l*self.count_c)                                                     
        cols = list(range(self.count_c))*l 
        data = [self.c_ft_value]*(l*self.count_c)
        c = sparse.csr_matrix((data, (rows, cols)))
        # build the probabilities to be multiplied by
        p = np.array(self.c_prob).reshape(-1,1)
        p = np.tile(p, (X.shape[0], 1))                                       
                                                                            
        # combine the original features and the possible confounder values    
        repeat_indices = np.arange(X.shape[0]).repeat(self.count_c)                      
        X = X[repeat_indices]      
        Xc = sparse.hstack((X,c)) 
        proba = self.clf.predict_proba(Xc)
        # multiply by P(z) and sum over the confounder for every instance in X
        proba *= p
        proba = proba.reshape(-1, self.count_c, self.count_y)
        proba = np.sum(proba, axis=1) 
        # normalize   
        norm = np.sum(proba, axis=1).reshape(-1,1)
        proba /= norm
        return proba                                                          
                                                                              
    def predict(self, X):                                                     
        proba = self.predict_proba(X)                                         
        return np.array(proba.argmax(axis=1))                                 
          
    def fit(self, X, y, c, c_ft_value=1.):
        self.c_prob = np.bincount(c)/len(c)                                            
        self.c_ft_value = c_ft_value
        self.count_c = len(set(c))
        self.count_y = len(set(y))

        rows = range(len(c))
        cols = c
        data = [c_ft_value]*len(c)
        c_fts = sparse.csr_matrix((data, (rows, cols)))
        Xc = sparse.hstack((X, c_fts))
        
        self.clf.fit(Xc, y)
        self.coef_=self.clf.coef_
        #self.coef_=np.array([self.clf.coef_[:][:-c_fts.shape[1]]])

def backdoor_adjustment_var_C(X, y, z, c, rand, feature_names):
    clf = BackdoorAdjustment()
    clf.fit(X, y, z, c_ft_value=c)
    return clf

ysize=2 # the number of y value
hmax=10 # the height threshold of tree
zsize=2
xsize=2
thresholdPA = 1.5
#causal decision tree
class CDecisionTree(object):
    def __init__(self):
        self.ysize=2
    def fit(self,X,Y,Z): #build tree
        height=0   # the initial height
        xshape=X.shape[1]
        yshape=1
        zshape=1
        self.xrows=list(range(X.shape[0]))
        ychange=np.reshape(Y,(Y.shape[0],yshape))
        zchange=np.reshape(Z,(Z.shape[0],zshape))
        hmatrix=sparse.hstack((X,ychange,zchange)).toarray()
        #self.data=sparse.csr_matrix(hmatrix[hmatrix[:,-1].argsort()])
        self.data=sparse.csr_matrix(hmatrix)
        self.yindex=-2
        self.zindex=-1
        ppp=PropensityScore(X[:,0].toarray(),Z)
        #print(ppp.compute())
        self.xindex=list(range(xshape))
        self.tree=self.treeConstruct(self.xindex,self.xrows,0)
        self.tree=self.treePrunning(self.tree)
    
    def get_feasize(self,X,xshape):
        size=[]
        xarray=X.toarray()
        for i in range(xshape):
            size.append(len(set(xarray[:,i])))
        return size
    
    def treeConstruct(self,X,D,h):
        h=h+1;
        if(D==[]):
            #print('data in none')
            return None
        if((X==[]) or (h==hmax)):
            label=self.popyvalue_select(D,self.yindex)
            leafnode=CDLeafNode(label)
            return leafnode
        dchange=self.data[D,-2].toarray().flatten().tolist()

        if(len(set(dchange))==1):
            label=list(set(dchange))[0]
            leafnode=CDLeafNode(label)
            return leafnode
        #select the most probability y value and return
        maxindex =X[0]
        #paValue=self.cal_PAMH(D,X[0])
        paValue=self.cal_PAMH(D,X[0])
        for index in X[1:]:
            #calculate the PAMH for each variable and find the max feature as the candiadate
            #cpa=self.cal_PAMH(D,index)
            cpa=self.cal_PAMH(D,index)
            if(cpa>paValue):
                paValue=cpa
                maxindex=index
        if(paValue<thresholdPA):
            label=self.popyvalue_select(D,-2)
            leafnode=CDLeafNode(label)
            return leafnode
        subset=self.split_subset(D,maxindex)
        X.remove(maxindex)
        nonode=CDTreeNode(h,maxindex)
        for i in range(2):
            node=self.treeConstruct(X,subset[i],h)
            nonode.child.append(node)
        return nonode
        
#split the data
    def split_subset(self,D,index):
        subset=[]
        for i in range(2):
            subset.append([])
        for i in D:
            subset[self.data[i,index]].append(i)
        return subset

#calculate the contengency table
    def cal_contTable(self,D,index):
        contable=np.zeros((zsize,ysize*2))
        for i in D:
            colindex=self.data[i,index]*2+self.data[i,self.yindex]
            rowindex=self.data[i,self.zindex]
            contable[rowindex,colindex]+=1
        return contable

    def cal_info_gain(self,D,index):
        
#calculate the PAMH  
    def cal_PAMH(self,D,index):
        upvalue=0
        downvalue=0
        contable=self.cal_contTable(D,index)
        for i in range(zsize):
            ma=contable[i,0]
            mb=contable[i,1]
            mc=contable[i,2]
            md=contable[i,3]
            sum=ma+mb+mc+md
            if(((ma+mb)*(ma+mc)*(mc+md)*(mb+md)!=0) and (mc+md)>=1):
                upvalue+=(ma*md-mb*mc)/sum
                downvalue+=(ma+mb)*(ma+mc)*(mc+md)*(mb+md)/(pow(sum,2)*(sum-1))
            else:
                upvalue+=0
                downvalue+=0
        if(downvalue==0):
            return 0
        paValue=pow((abs(upvalue)-0.5),2)/downvalue
        return paValue
    def cal_propensity(self,D,index):
        ppp=PropensityScore(self.data[D,index].toarray(),self.data[D,-1].toarray())
        table=ppp.compute()
        hmatrix=sparse.hstack((self.data[D,:],np.reshape(table,(len(table),1)))).toarray()
        hdata=hmatrix[hmatrix[:,-1].argsort()]
        value=hdata[0,-1]
        con_table=[]
        element=np.zeros(4)
        con_table.append(element)
        j=0
        for i in hdata:
            #print(type(i))
            #print(i.shape)
            if(i[-1]==value):
                #print(con_table)
                con_table[j][int(i[index]*2+i[-3])]+=1
            else:
                value=i[-1]
                element=np.zeros(4)
                con_table.append(element)
                #print(con_table)
                j=j+1
                con_table[j][int(i[index]*2+i[-3])]+=1
        return con_table
        
    def cal_PAMH2(self,D,index):
        upvalue=0
        downvalue=0
        contable=self.cal_propensity(D,index)
        for i in range(len(contable)):
            ma=contable[i][0]
            mb=contable[i][1]
            mc=contable[i][2]
            md=contable[i][3]
            sum=ma+mb+mc+md
            if(((ma+mb)*(ma+mc)*(mc+md)*(mb+md)!=0) and (mc+md)>=1):
                upvalue+=(ma*md-mb*mc)/sum
                downvalue+=(ma+mb)*(ma+mc)*(mc+md)*(mb+md)/(pow(sum,2)*(sum-1))
            else:
                upvalue+=0
                downvalue+=0
        if(downvalue==0):
            return 0
        paValue=pow((abs(upvalue)-0.5),2)/downvalue
        return paValue
#D: the index array of subdataset
#Nindex: the index of controlled feature
#Nvalue: the value of controlled feature'''
    def popyvalue_select(self,D,nindex):
        valFreq = {}
        for tuple in D:
            if(valFreq.__contains__(self.data[tuple,nindex])):
                valFreq[self.data[tuple,nindex]] += 1
            else:
                valFreq[self.data[tuple,nindex]] = 1
        max = -1
        major = ""
        for key in valFreq.keys():
            if valFreq[key]>max:
                max = valFreq[key]
                major = key
        return major
    
    def treePrunning(self,node):
        if((type(node)==CDLeafNode) or(node==None)):
            return node
        for i in range(len(node.child)):
            if(type(node.child[i])==CDTreeNode):
                node.child[i]=self.treePrunning(node.child[i])
            if(node.child[i]==None):
                node.child[i]=node.child[1-i]        
        return node
    
    def predict(self, D):
        proba=np.zeros(D.shape[0])
        self.datatest=D
        self.predict_proba(list(range(D.shape[0])),proba,self.tree)
        return proba
    def predict_proba(self,D,proba,node):
        if(D==[]):
            return
        if(type(node)==CDLeafNode):
            for i in D:
                proba[i]=node.ylabel
            return
        subset=self.split(D,node.feature)
        for i in range(2):
            self.predict_proba(subset[i],proba,node.child[i])
    def split(self,D,index):
        subset=[]
        for i in range(2):
            subset.append([])
        for i in D:
            subset[self.datatest[i,index]].append(i)
        return subset
        
        

#non-leaf node        
class CDTreeNode(object):
    def __init__(self,h,feaindex):
        self.height=h
        self.feature=feaindex
        self.child=[]
        
#leafNode         
class CDLeafNode(object):
    def __init__(self,label):
        #self.height=h;
        self.ylabel=label
        
def cdtAlg(X, y, z, rand, feature_names):
    clf= CDecisionTree()
    clf.fit(X,y,z)
    return clf


class PropensityScore(object):
    """
    Estimate the propensity score for each observation.
    
    The compute method uses a generalized linear model to regress treatment on covariates to estimate the propensity score. 
    This is not the only way to estimate the propensity score, but it is the most common.
    The two options allowed are logistic regression and probit regression.
    """

    def __init__(self, treatment, covariates):
        """
        Parameters
        -----------        
        treatment : array-like
            binary treatment assignment
        covariates : pd.DataFrame
            covariates, one row for each observation
        """
        assert treatment.shape[0]==covariates.shape[0], 'Number of observations in \
            treated and covariates doesnt match'
        self.treatment = treatment
        self.covariates = covariates
        
    def compute(self, method='logistic'):
        """
        Compute propensity score and measures of goodness-of-fit
        
        Parameters
        ----------
        method : str
            Propensity score estimation method. Either 'logistic' or 'probit'
        """
        predictors = sm.add_constant(self.covariates, prepend=False)
        if method == 'logistic':
            model = sm.Logit(self.treatment, predictors).fit(method="bfgs",disp=False, warn_convergence=False)
        elif method == 'probit':
            model = sm.Probit(self.treatment, predictors).fit(disp=False, warn_convergence=True)
        else:
            raise ValueError('Unrecognized method')
        return model.predict()