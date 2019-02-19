from sklearn.feature_selection import SelectKBest, mutual_info_classif as IG
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics import accuracy_score as acc
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier as AB, RandomForestClassifier as RF

def Process(X, y, rtype = 'number', k = 10, encoded = False, svm_kernel = 'rbf'):
    '''
    Parameters:
        X: (required) numpy ndarray
            Array of floating point numbers
            Confined to be 2D, with the shape as (<Number of samples>,<Number of Features>)
        y: (required) numpy ndarray
            Array of strings 
            Confined to be 1D, with the shape as (<Number of samples>,)
        r_type: string
            Either 'number' or 'percent'
            Corresponds to 
        k: integer
            Cs
        encoded: boolean
            If the labels are already encoded as integers
        svm_kernel: string
            Selected kernel of SVM classifier
    '''
    
    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert X.shape[0] == y.shape[0]
    print('########################################')
    print('Selected SVM kernel is', svm_kernel)
    
    num_samples = X.shape[0]
    num_features = X.shape[1]
    print('Number of samples:', num_samples)
    print('Number of features:', num_features)
    print('########################################')
    
    if encoded:
        print('Labels are already encoded as integers.')
    else:
        print('Labels are not encoded. Call LabelEncoder.')
        encoder = LE()
        y = encoder.fit_transform(y)
        print('Labels are encoded as integers.')
    print('########################################')
    
    if rtype == 'number':
        FR = SelectKBest(IG, k).fit(X, y) # Feature Reduction with Information Gain
    elif rtype == 'percent':
        FR = SelectKBest(IG, k).fit(X, y) # Feature Reduction with Information Gain
    else:
        raise ValueError('Reduction type unsupported')
    
    trainX = FR.transform(X[:num_samples // 2])
    trainY = y[:num_samples // 2]
    testX = FR.transform(X[num_samples // 2:])
    testY = y[num_samples // 2:]
    print('Training feature shape:', trainX.shape)
    print('Test feature shape', testX.shape)
    print('########################################')
    
    svmclf = SVC(gamma = 'scale', kernel = svm_kernel) # SVM classifier
    _ = svmclf.fit(trainX, trainY)
    ABclf = AB() # AdaBoost classifier
    _ = ABclf.fit(trainX, trainY)
    RFclf = RF(n_estimators = 100, max_depth = 30, random_state = 0) # Random Forest classifier
    _ = RFclf.fit(trainX, trainY)
    
    svmres = svmclf.predict(testX) # SVM withresult
    ABres = ABclf.predict(testX) # Adaboost result
    RFres = RFclf.predict(testX) # Random Forest result
    
    svmscore = acc(testY, svmres)
    ABscore = acc(testY, ABres)
    RFscore = acc(testY, RFres)
    print('Accuracy of SVM is', '{:.4f}'.format(svmscore))
    print('Accuracy of AdaBoost is', '{:.4f}'.format(ABscore))
    print('Accuracy of Random Forest is', '{:.4f}'.format(RFscore))
    print('########################################')
    
    return None

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    digits = load_digits()
    num = digits.target.shape[0]
    X = digits.images.reshape(num, -1)
    y = digits.target
    Process(X, y, encoded = True)