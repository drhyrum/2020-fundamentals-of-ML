# conda create -n MLclass python=3.7 scikit-learn pandas numpy ipython jupyter[notebook]
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

data = pd.read_csv('../../data/20questions/micro.csv')

print(data)

target = data[data.columns[0]]
feature_names = data.columns[1:]
features = data[feature_names]

model = DecisionTreeClassifier().fit(features, target)