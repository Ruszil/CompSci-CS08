def calculate_prior(data, class_label):
    total_instances = len(data)
    class_instances = sum(1 for row in data if row[-1] == class_label)
    return class_instances / total_instances

def calculate_likelihood(data, feature_index, feature_value, class_label):
    class_instances = sum(1 for row in data if row[-1] == class_label)
    feature_matches = sum(1 for row in data if row[-1] == class_label and row[feature_index] == feature_value)
    return feature_matches / class_instances if class_instances else 0  # Handle zero division

def naive_bayes_predict(data, input_features, prior_probs, likelihood_probs):
    classes = sorted(list(set(row[-1] for row in data)))
    posterior_probs = {}
    for class_label in classes:
        posterior_probs[class_label] = prior_probs[class_label]
        for i, feature_value in enumerate(input_features):
            posterior_probs[class_label] *= likelihood_probs[i][feature_value][class_label]
    return max(posterior_probs, key=posterior_probs.get)
data = [
    ['Red', 'Sports', 'Domestic', 'Yes'],
    ['Red', 'Sports', 'Domestic', 'No'],
    ['Red', 'Sports', 'Domestic', 'Yes'],
    ['Yellow', 'Sports', 'Domestic', 'No'],
    ['Yellow', 'Sports', 'Imported', 'Yes'],
    ['Yellow', 'SUV', 'Imported', 'No'],
    ['Yellow', 'SUV', 'Imported', 'Yes'],
    ['Yellow', 'SUV', 'Domestic', 'No'],
    ['Red', 'SUV', 'Imported', 'No'],
    ['Red', 'Sports', 'Imported', 'Yes'],
]

prior_probs = {
    'Yes': calculate_prior(data, 'Yes'),
    'No': calculate_prior(data, 'No'),
}

likelihood_probs = []
for i in range(3): # For Color, Type, Origin
    likelihood_probs.append({})
    for row in data:
        feature_value = row[i]
        class_label = row[-1]
        if feature_value not in likelihood_probs[i]:
            likelihood_probs[i][feature_value] = {}
        likelihood_probs[i][feature_value][class_label] = likelihood_probs[i][feature_value].get(class_label,0) + 1

for i in range(3):
    for feature_value in likelihood_probs[i]:
        for class_label in ['Yes', 'No']:
            likelihood_probs[i][feature_value][class_label] = calculate_likelihood(data, i, feature_value, class_label)

input_features = ['Yellow', 'SUV', 'Imported']
prediction = naive_bayes_predict(data, input_features, prior_probs, likelihood_probs)
print(f"Prediction for {input_features}: Stolen = {prediction}")


input_features = ['Red', 'Sports', 'Domestic']
prediction = naive_bayes_predict(data, input_features, prior_probs, likelihood_probs)
print(f"Prediction for {input_features}: Stolen = {prediction}")
