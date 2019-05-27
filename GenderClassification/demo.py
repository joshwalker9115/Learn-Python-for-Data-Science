from sklearn import tree
from sklearn import ensemble
from sklearn import svm

clf0 = tree.DecisionTreeClassifier()
# CHALLENGE - create 3 more classifiers...
# 1
clf1 = ensemble.RandomForestClassifier(n_estimators=10)
# 2
clf2 = ensemble.AdaBoostClassifier(n_estimators=10)
# 3
clf3 = svm.SVC(gamma='scale')

#height, weight, shoe size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
    [190, 90, 47], [175, 64, 39],
    [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

# CHALLENGE - ...and train them on our data
clf0 = clf0.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)
clf3 = clf3.fit(X, Y)

#True/ False comparison count
prediction0 = sum(bool(x) for x in (clf0.predict(X) == Y))
prediction1 = sum(bool(x) for x in (clf1.predict(X) == Y))
prediction2 = sum(bool(x) for x in (clf2.predict(X) == Y))
prediction3 = sum(bool(x) for x in (clf3.predict(X) == Y))

#Display results
print("Decision Tree Results: ",prediction0,"/11 ",(clf0.predict(X) == Y))
print("Random Forest Results: ",prediction1,"/11 ",(clf1.predict(X) == Y))
print("AdaBoost Results: ",prediction2,"/11 ",(clf2.predict(X) == Y))
print("Support Vector Machine Results: ",prediction3,"/11 ",(clf3.predict(X) == Y))