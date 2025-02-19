from joblib import load
from sklearn.tree import DecisionTreeClassifier

# Load the model
model_name = 'decision_tree'
model_path = f'./saved_model/{model_name}.joblib'
clf = load(model_path)

# Display basic model information
print("Loaded model details:")
print(clf)

# If the model is a DecisionTreeClassifier, print out specific attributes
if isinstance(clf, DecisionTreeClassifier):
    print("\nModel Parameters:")
    print(clf.get_params())

    print("\nFeature Importances:")
    print(clf.feature_importances_)

    print("\nTree Structure:")
    print("Number of features: ", clf.n_features_in_)
    print("Number of outputs: ", clf.n_outputs_)
    print("Number of nodes: ", clf.tree_.node_count)

    # You can also print out the structure of the tree (optional)
    from sklearn import tree
    tree_rules = tree.export_text(clf)
    print("\nTree Rules:\n")
    print(tree_rules)

# For other model types (e.g., RandomForestClassifier, GradientBoostingClassifier), you can adjust the code accordingly
elif isinstance(clf, RandomForestClassifier):
    print("\nModel Parameters:")
    print(clf.get_params())

    print("\nFeature Importances:")
    print(clf.feature_importances_)

    print("\nEstimators:")
    print(clf.estimators_)

# Add additional checks for other model types if needed

# General model attributes
print("\nClasses:")
print(clf.classes_)

print("\nNumber of Classes:")
print(clf.n_classes_)
