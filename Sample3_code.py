"""
bioResponse.py: Predict a biological response of molecules from their chemical properties

Created on Wed Jul 29 21:50:33 2015

@author: wuuyue@gmail.com

Outline
1) split the data to train set - 5 parts = 80% and test set - 20% (reserved test set)

2) Second, we will run the code below to build three types of models
          
               a ETC model: extra tree classification
               a SVM model: support vector machine 
               a RFC model: random forest classification
           
3) Last, using the splitted test set, we will computer the distances of the 
         predicted probability and the true value (20% preserved) for each model, 
         and plot all of them in one subplot for comparison.
"""

import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import math
import csv_io

#define the RMS metric between two list
def prdmtrc(xPred, xTgrt):
    d = 0 
    n1 = len(xPred)
    n2 = len(xTgrt)
    n = min(n1, n2)
    for i in range(n):
        d += (abs(float(xPred[i])-float(xTgrt[i])))**2
        continue
    try:
        davg = math.sqrt(d/n)
        return davg
    except ZeroDivisionError:
        print("An Error occured")
        print("due to zero division")
#model building and metric computation
def main():
    prts = input("How many parts(>=5) do you want?\n")
    datain = csv_io.read_data("./train.csv")
    target_all=[x[0] for x in datain]
    data_all = [x[1:] for x in datain]
    total=len(datain)
    notr=int(total/float(prts))
    target_tr = target_all[:4*notr]
    target_tt = target_all[4*notr:5*notr]
    target_tt_f= ["%f" %x for x in target_tt]
    data_tr = data_all[:4*notr]
    data_tt = data_all[4*notr:5*notr]
    print ("selected %i out of %i \n" % (4*notr,total) )
    print ("length for each sample data: %f\n" % len(data_tr[notr-1]))
    #    for item in data_tr[notr-1]:
    #        print(item)
        
    test = csv_io.read_data("./test.csv") #provided test data set
    
    #Create Extra Tree Classification model
    etc = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
    etc.fit(data_tr, target_tr)
    
     #generate and save the SVM prediction for the provided test data
    pred_probs0_etc1 = etc.predict_proba(test)
    print (len(pred_probs0_etc1), pred_probs0_etc1[:])
    pred_probs0_etc1 = ["%f" % x[1] for x in pred_probs0_etc1]
    csv_io.write_delimited_file("./rf_benchmark21etc1.csv", pred_probs0_etc1)
    
    #generate the prediction for the splitted test data
    #compute the RMS distance to the true value-SVM model
    pred_probs_etc1 = etc.predict_proba(data_tt)
    #print (len(pred_probs_svm1), pred_probs_svm1[:])
    pred_probs_etc1 = ["%f" % x[1] for x in pred_probs_etc1]
    metric_etc1 = prdmtrc(pred_probs_etc1,target_tt_f)
    print (" ********    The RMS distance between ETC1 prediction\n")
    print (" ********    and the targets on the reserved test set is: %f" % metric_etc1)
    
    #create SVM model
    clfSVM = svm.SVC(probability=True)
    svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\
            decision_function_shape=None, degree=3, gamma='auto', \
            kernel='rbf', max_iter=-1, probability=True, \
            random_state=None, shrinking=True,\
            tol=0.0001, verbose=True)    
    clfSVM.fit(data_tr, target_tr)
    
        
    #generate and save the SVM prediction for the provided test data
    pred_probs0_svm1 = clfSVM.predict_proba(test)
    print (len(pred_probs0_svm1), pred_probs0_svm1[:])
    pred_probs0_svm1 = ["%f" % x[1] for x in pred_probs0_svm1]
    csv_io.write_delimited_file("./rf_benchmark21svm1.csv", pred_probs0_svm1)
    
    #generate the prediction for the splitted test data
    #compute the RMS distance to the true value-SVM model
    pred_probs_svm1 = clfSVM.predict_proba(data_tt)
    #print (len(pred_probs_svm1), pred_probs_svm1[:])
    pred_probs_svm1 = ["%f" % x[1] for x in pred_probs_svm1]
    metric_svm1 = prdmtrc(pred_probs_svm1,target_tt_f)
    print (" ********    The RMS distance between SVM1 prediction\n")
    print (" ********    and the targets on the reserved test set is: %f" % metric_svm1)
    
    #random forest method
    rfc1 = RandomForestClassifier(n_estimators=100)
    rfc1.fit(data_tr, target_tr)
    
    #generate and save the RFC prediction for the provided test data
    pred_probs0_rfc1 = rfc1.predict_proba(test)
    pred_probs0_rfc1 = ["%f" % x[1] for x in pred_probs0_rfc1]
    csv_io.write_delimited_file("./rf_benchmark21rfc1.csv", pred_probs0_rfc1)

    #generate the prediction for the splitted test data
    #compute the RMS distance to the true value-random forest model
    pred_probs_rfc1 = rfc1.predict_proba(data_tt)
    #print (len(pred_probs_svm1), pred_probs_svm1[:])
    pred_probs_rfc1 = ["%f" % x[1] for x in pred_probs_rfc1]
    metric_rfc1 = prdmtrc(pred_probs_rfc1,target_tt_f)
    print ("---------------------------------------------------------")
    print (" ********    The RMS distance between RFC1 prediction\n")
    print (" ********    and the targets on the reserved test set is: %f" % metric_rfc1)
    print ("---------------------------------------------------------")
    #Plot the distances vs the corresponding models 
    # boxplot algorithm comparison
    names = []
    names.append('ETC')
    names.append('SVM')
    names.append('RFC')
 
    results=[]
    results.append(metric_etc1)   
    results.append(metric_svm1)
    results.append(metric_rfc1)
 
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison (the smaller metric the better)')
    ax = fig.add_subplot(111)
    print (results, names)
    plt.xticks([1,2,3],names)
    plt.plot([1,2,3],results, 'ro')
    #ax.set_xticklabels(names)
    plt.show()
    
#run      
if __name__=="__main__":
    main()
