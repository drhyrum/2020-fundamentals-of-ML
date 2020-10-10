import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# ####### PART 1: READ IN AND PREPARE DATA

# read file from disk
df = pd.read_csv('../../data/icecream/survey.csv')

# fix a typo (MY FAULT!)
keys = list(df.keys())
df.loc[0, keys[-2]] = 'waffle cone/bowl'

# turn data into numbers
ismale = (np.atleast_2d(df[keys[2]]).T == 'Male').astype(np.float32)
age = np.atleast_2d(df[keys[3]]).T.astype(np.float32)

snack_ohe = preprocessing.OneHotEncoder(sparse=False)
snack = snack_ohe.fit_transform(np.atleast_2d(df[keys[4]]).T)

icecream_ohe = preprocessing.OneHotEncoder(sparse=False)
icecream = icecream_ohe.fit_transform(np.atleast_2d(df[keys[5]]).T)

cone_ohe = preprocessing.OneHotEncoder(sparse=False)
cone = cone_ohe.fit_transform(np.atleast_2d(df[keys[7]]).T)

# design decision: choosing to make "scoops" a categorical variable instead of a raw number
#    ...because WE'LL NEVER GIVE YOU 4 SCOOPS!
scoops_ohe = preprocessing.OneHotEncoder(sparse=False)
scoops = scoops_ohe.fit_transform(np.atleast_2d(df[keys[8]]).T)

# toppings is a comma-separated string "caramel, hot fudge, nuts, whipped cream"
# first, turn it into a list of strings: "caramel", "hot fudge", "nuts", "whipped cream"
toppings_list = [t.split(", ") for t in df[keys[6]]]
toppings_mlb = preprocessing.MultiLabelBinarizer()  # toppings_mlb.classes_
toppings = toppings_mlb.fit_transform(toppings_list).astype(np.float32)

# set up inputs X and outputs (targets) Y
Y = toppings
topping_names = toppings_mlb.classes_

X = np.concatenate([ismale, age, snack, icecream, cone, scoops], axis=1)
feature_names = ['gender'] + ['age']
feature_names += ['snack-' + s for s in snack_ohe.categories_[0]]
feature_names += ['icecream-' + s for s in icecream_ohe.categories_[0]]
feature_names += ['cone-' + s for s in cone_ohe.categories_[0]]
feature_names += ['scoops-' + str(s) for s in scoops_ohe.categories_[0]]

# ####### PART 2: TRAIN A MODEL

# holdout some data to validate that it works (IMPORTANT!)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=20)

# train a model on the data
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# ####### PART 3: TEST MODEL'S PERFORMANCE

# validate that it works on the holdout set.  Does it overfit to training?
Y_pred_proba = np.array(model.predict_proba(X_val))[..., 1].squeeze().T
Y_pred = (Y_pred_proba > 0.5).astype(float)

# how did we do on the holdout set?
print("\nAccuracy for each topping:")
for i, n in enumerate(topping_names):
    score = accuracy_score(Y_val[:, i], Y_pred[:, i]) * 100
    print(f"{i+1}. {n} accuracy = {score}%")

# which features were most important?
print("\nImportant features:")
for i, (score, name) in enumerate(sorted(list(zip(model.feature_importances_, feature_names)), reverse=True)):
    print(f"{i+1}. {name} ({score})")


# ####### PART 4: SAVE MODEL TO DISK

# let's save everything we need
toppings_predictor = {
    'model': model,
    'scoops_ohe': scoops_ohe,
    'toppings_mlb': toppings_mlb,
    'cone_ohe': cone_ohe,
    'icecream_ohe': icecream_ohe,
    'snack_ohe': snack_ohe,
}

pickle.dump(toppings_predictor, open('toppings_predictor.pkl', 'wb'))
