from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np
import itertools


# load the Iris dataset and print it
df = load_iris(as_frame=True)['frame']
print(df)

# reduce to just 2 dimensions and 2 kinds of flowers (keep the problem simple for illustration)
df = df[['sepal length (cm)', 'petal length (cm)', 'target']]
df = df[df['target'].isin([1, 2])]

# plot the data
f1, ax1 = plt.subplots(1, 2)
ax1[0].scatter(df['sepal length (cm)'], df['petal length (cm)'], c=df['target'])
ax1[0].set_xlabel('sepal length (cm)')
ax1[0].set_ylabel('petal length (cm)')


# fit a decision tree model to the data
model = DecisionTreeClassifier(criterion='entropy', max_depth=2)
model.fit(X=df[['sepal length (cm)', 'petal length (cm)']].to_numpy(), y=df['target'])

# plot the learned decision tree
plt.figure()
plot_tree(model, fontsize=10, feature_names=['sepal length (cm)', 'petal length (cm)'])

# visualize how the model divides up the space into two classes
# by using it to classify test points that span the space
x_test_pts = np.linspace(df.iloc[:, 0].min(), df.iloc[:, 0].max(), 50)
y_test_pts = np.linspace(df.iloc[:, 1].min(), df.iloc[:, 1].max(), 50)
test_coords = np.array(list(itertools.product(x_test_pts, y_test_pts)))
classifications = [model.predict_proba([test_coord])[0][0] for test_coord in test_coords]

ax1[1].scatter(test_coords[:,0], test_coords[:, 1], c=classifications)

plt.show()
