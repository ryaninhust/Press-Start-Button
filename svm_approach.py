from sklearn import svm
from sklearn.datasets import load_svmlight_files

X_train, y_train, X_test, y_test = load_svmlight_files((
    'data/ml14fall_train.dat', 'data/ml14fall_test1_no_answer.dat'))

print "read data finished"

poly_clf = svm.SVC(kernel='poly', degree=5)
poly_clf = poly_clf.fit(X_train[:50], y_train[:50])
print "fit model finished"
prediction = poly_clf.predict(X_test[:50])
print prediction

def write_result(pred, result_path):
    result_content = '\n'.join([str(int(p)) for p in pred])
    with open(result_path, 'w') as result:
        result.write(result_content)
    print "result has saved into %s" % result_path

write_result(prediction, 'poly_5_path')



