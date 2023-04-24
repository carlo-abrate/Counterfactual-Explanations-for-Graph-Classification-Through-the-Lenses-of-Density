parameters = {"SF": {"model_number":(4,1)},"contrast": {"model_number":(4,1)}
    
    # for each classifier_name, put the corresponding classifier
}

SEED = 42

import networkx as nx
import numpy as np
from karateclub import SF
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import random

random.seed(SEED)

class Oracle:

    def __init__(self, method, data=[],idc=0):
        self.method = method
        self.data = data
        self.idc = idc
        
    def embedding_phase(self):
        """
        """
        #
        if self.method=="contrast":
            yx0 = [(y,contrast_feature_extraction(g)) for g,y in self.data if y==0]
            yx1 = [(y,contrast_feature_extraction(g)) for g,y in self.data if y==1]
            return yx0, yx1, []
        elif self.method=="SF":
            gs = [g for g,y in self.data]
            ys = [y for g,y in self.data]
            #
            embed_model = SF(dimensions=200,seed=42)
            #
            embed_model.fit(gs)
            vectors = embed_model.get_embedding()
            yx0 = [(y,x) for x,y in zip(vectors,ys) if y==0]
            yx1 = [(y,x) for x,y in zip(vectors,ys) if y==1]
            return yx0, yx1, embed_model
    
        
    def train(self):
        yx0, yx1, self.embed_model = self.embedding_phase()
        results,models_classifier = training_function(yx0, yx1)
        r_acc = [(v[0],k) for k,v in results.items()]
        mod_id = self.idc
        model_order = sorted(r_acc, key=lambda element: element[0], reverse=True)
        print(model_order)
        print(models_classifier)
        #id_model= model_order[mod_id[0]][mod_id[1]]
        id_model = model_order[self.idc][1]
        #print(id_model, r_acc)
        self.classifier = models_classifier[id_model][0]
        print(self.classifier)
        return self

    def predict(self, g):
        if self.method=="contrast":
            #g_input = nx.convert_matrix.to_numpy_array(g)
            f = contrast_feature_extraction(g)
            return self.classifier.predict([f])[0]
        elif self.method=="SF":
            self.embed_model.fit([g])
            g_v = self.embed_model.get_embedding()[0]
            y_hat = self.classifier.predict([g_v])[0]
            return y_hat
        
    def evaluate_classifier(self,test_data):
        """
        """
        r = []
        for g,y in test_data:
            r.append((self.predict(g),y))
        ys_true = [i[1] for i in r]
        ys_hat = [i[0] for i in r]
        tn, fp, fn, tp = confusion_matrix(ys_true,ys_hat).ravel()
        print('Results:\n- {} TP;\n- {} TN;\n- {} FP;\n- {} FN.'.format(tp,tn,fp,fn))
        accuracy = accuracy_score(ys_true,ys_hat)
        print('Accuracy = {}'.format(accuracy))
        print("F1 = ", f1_score(ys_true, ys_hat))
                    
    
    
    
    
def sub_graph(g,v_sub):
    '''To create the sub graph og 'g' from the list of nodes in 'v_sub'.
    '''
    g_sub = np.copy(g)
    #l_1 = [el for el in v_sub]
    l_1 = [el for el in v_sub]
    g_sub = g_sub[np.ix_(l_1,l_1)]
    return g_sub

def contrast_feature_extraction(g_raw):
    ''' The classification funcion for the graph 'g'
    '''
    # Sub-graphs
    g = nx.convert_matrix.to_numpy_array(g_raw)
    td_asd = [65, 70, 99, 80, 69, 6, 7, 8, 9, 13, 77, 45, 16, 81, 78, 92, 56, 57, 60, 93, 63]
    asd_td = [0, 36, 37, 38, 81, 40, 41, 74, 75, 76, 70, 72, 114, 20, 21, 73, 90, 28, 29]

    # Induced sub-graphs
    g_td_asd = sub_graph(g,td_asd)
    g_asd_td = sub_graph(g,asd_td)

    # Coefficients
    a = sum([sum(i) for i in g_td_asd])/2
    b = sum([sum(i) for i in g_asd_td])/2
    return [a,b]



def classifiers_test(x_train,y_train,x_test,y_test):
    '''
    '''
    results_i = []
    models = []
    
    # KNN
    training_accuracy = [] 
    test_accuracy = []
    kn_models = []
    neighbors_settings = range(1, 30)
    for n_neighbors in neighbors_settings:
        # build the model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(x_train, y_train)
        # record training set accuracy 
        training_accuracy.append(clf.score(x_train, y_train)) 
        # record generalization accuracy 
        scc = clf.score(x_test, y_test)
        test_accuracy.append(scc)
        kn_models.append(clf)
    kn = test_accuracy.index(max(test_accuracy))
    results_i.append(test_accuracy[kn])
    models.append(kn_models[kn])
    
    # SCM
    test_accuracy_svm = []
    clf = svm.SVC(kernel='linear', C = 1.0)
    clf.fit(x_train, y_train)
    results_i.append(clf.score(x_test, y_test))
    models.append(clf)
    
    return results_i,models


def training_function(yx0, yx1):
    """
    """
    random.shuffle(yx0)
    random.shuffle(yx1)
    lenmax = max(len(yx0),len(yx1))
    yx0 = yx0[:lenmax]
    yx1 = yx1[:lenmax]
    yx = yx0 + yx1
    random.shuffle(yx)
    X = np.array([el[1] for el in yx])
    Y = np.array([el[0] for el in yx])
    ##
    kf = KFold(n_splits=5,shuffle=True,random_state=(SEED))
    kf.get_n_splits(X)
    i = 0
    results = {}
    models_classifier = {}
    for train_index, test_index in kf.split(X):
        train_index = list(train_index)
        test_index = list(test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        #print(len(X_train),len(X_test),len(y_train),len(y_test))
        # Classifiers
        results[i],models_classifier[i] = classifiers_test(X_train,y_train,X_test,y_test)
        i+=1
    return results,models_classifier
