import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('customer_purchase_data_colab.csv')
X = dataset.iloc[:, [1,2]].values
y = dataset.iloc[:, 3].values

print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


import seaborn as sns
plt.figure(figsize=(8,6))

sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette={0: 'blue', 1: 'red'}, marker = 'o')

plt.xlabel("Age")
plt.ylabel("Estimated salary")

plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)

import seaborn as sns
plt.figure(figsize=(8,6))

sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette={0: 'blue', 1: 'red'}, marker = 'o')

plt.xlabel("Age")
plt.ylabel("Estimated salary")

plt.show()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, confusion_matrix

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%" .format(accuracy * 100))

print("Coefficients:", classifier.coef_)
print("Intercept:", classifier.intercept_)

import seaborn as sns
x1 = np.linspace(-3, 3, 100)
x2 = (-2.07665837* x1 +0.95217247)/1.11008221
plt.figure(figsize=(8,6))
plt.plot(x1, x2, color='green')
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette={0: 'blue', 1: 'red'}, marker = 'o')

plt.xlabel("Age")
plt.ylabel("Estimated salary")

plt.show()

We can dump this on VSDSQUADPRO board
Can predict if it will rain or not based on humidity and temperature


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(f"Test Accuracy (k=5) :{accuracy_score(y_test, y_pred):.2f}")

x1_vals = np.linspace(-3, 3, 400)
x2_vals = np.linspace(-3, 3, 400)
x1,x2 = np.meshgrid(x1_vals, x2_vals)

Z = knn.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)

print(Z)
import seaborn as sns
plt.figure(figsize=(8,6))
plt.contour(x1, x2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
sns.scatterplot(x=X_train[:, 0], y=X_train[:, 1], hue=y_train, palette={0: 'blue', 1: 'red'}, marker = 'o')

plt.xlabel("Age")
plt.ylabel("Estimated salary")

plt.show()

(For knn. Its showing as knnx in above codes)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0) //change linear to ‘rbf’ and check accuracy and result, accuracy would be increased to 93%
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() + 1, step =0.01),
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 0].max() + 1, step =0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]). T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red' , 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

print(classifier.coef_) //change to classifier.dual_coeff_ and check results
print(classifier.intercept_


with open("svm_model.h", "w") as f:
  f.write(f"#define NUM_CLASSES {weights.shape[0]}\n")
  f.write(f"#define NUM_FEATURES {weights.shape[1]}\n\n")

  f.write("double weights[NUM_CLASSES][NUM_FEATURES] {\n")
  for row in weights:
    f.write("  {" + ", ".join(f"{v:.10f}" for v in row) + "}, \n")
  f.write("}; \n\n")

  f.write("double bias[NUM_CLASSES] = {" + ", ".join(f"{b:.10f}" for b in bias) + "};\n")

print("Exported SVM model to svm_model.h")

Mean = sc.mean_
Scale =sc.scale_
F.write(f”#define NUM_FEATURES {len(mean)}\n\n”)
F.write(“double mean[NUM_FEATURES] = {\n”)
F.write(“ “ + “, “.join(f”{m:.10f}” for m in mean) + “\n};\n\n”)

F.write(“double scale[NUM_FEATURES] = {\n”)
f.write(“ “ + “, “.join(f”{s:.10f}” for s in scale) + “\n};\n”)

print(“exported scalar parameters to scalar.h”)
