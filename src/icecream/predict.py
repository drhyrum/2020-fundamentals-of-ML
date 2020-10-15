import pickle
import numpy as np

toppings_predictor = pickle.load(open('toppings_predictor.pkl', 'rb'))

model = toppings_predictor['model']
scoops_ohe = toppings_predictor['scoops_ohe']
toppings_mlb = toppings_predictor['toppings_mlb']
cone_ohe = toppings_predictor['cone_ohe']
icecream_ohe = toppings_predictor['icecream_ohe']
snack_ohe = toppings_predictor['snack_ohe']


def get_int_between(low, high):
    while True:
        selection = input(": ")
        try:
            selection = int(selection)
        except ValueError:
            continue
        if selection < low or selection > high:
            continue
        return selection


def ask(question, options):
    print(question)
    print(f'Please select one of the following:')
    for i, opt in enumerate(options):
        n = i + 1
        print(f'({n}) {opt}')

    selection = get_int_between(1, len(options)) - 1
    return options[selection]


print("Please enter your age (between 0 and 100)")
age = get_int_between(0, 100)
gender_s = ask("What is your gender?", ["Male", "Female"])
snack_s = ask("How would you best describe your snack preferences?", snack_ohe.categories_[0])
icecream_s = ask("Of the following options, which is your favorite ice cream flavor?", icecream_ohe.categories_[0])
cone_s = ask("Which cup or cone/bowl?", cone_ohe.categories_[0])
scoops = ask("How many scoops?", scoops_ohe.categories_[0])

# convert to numbers
ismale = np.array([float(gender_s == 'Male')])
age = np.array([age])
snack = snack_ohe.transform([[snack_s]])[0]
icecream = icecream_ohe.transform([[icecream_s]])[0]
cone = cone_ohe.transform([[cone_s]])[0]
scoops = scoops_ohe.transform([[scoops]])[0]

x_test = np.concatenate([ismale, age, snack, icecream, cone, scoops], axis=0)

y_pred_prob = np.array(model.predict_proba([x_test])).squeeze()[:, 1]
y_pred = y_pred_prob > 0.25
topping_names = toppings_mlb.classes_[y_pred]
topping_proba = y_pred_prob[y_pred]
print("You should try the following toppings (in order):")
for i, (prob, name) in enumerate(sorted(zip(topping_proba, topping_names), reverse=True)):
    print(f"{i+1}. {name} ({round(prob*100)}%)")
