import json

with open('output-v8-trainval.json') as f:
    data = json.load(f)

correct_predictions = 0
total_predictions = 0

for prediction in data:
    image_path = prediction['image_path']
    max_prob_class_name = prediction['max_prob_class_name']
    
    if max_prob_class_name.lower() in image_path.lower():
        correct_predictions += 1
    total_predictions += 1

accuracy = correct_predictions / total_predictions
print(f'Accuracy: {accuracy:.2f}')
print(f'Correct predictions: {correct_predictions}')
print(f'Total predictions: {total_predictions}')


# v8 - val
# Accuracy: 0.60
# Correct predictions: 503
# Total predictions: 837

#  v11 - val
# Accuracy: 0.61
# Correct predictions: 507
# Total predictions: 837

#  v11 - train
# Accuracy: 0.72
# Correct predictions: 9708
# Total predictions: 13491

# best2 - val
# Accuracy: 0.62
# Correct predictions: 520
# Total predictions: 837

# final combined - yolo v11
# Accuracy: 0.76
# Correct predictions: 10908
# Total predictions: 14328

# final combined - yolo v8
# Accuracy: 0.62
# Correct predictions: 8819
# Total predictions: 14328


precision = correct_predictions / total_predictions
