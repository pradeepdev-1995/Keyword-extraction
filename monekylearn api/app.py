from monkeylearn import MonkeyLearn

ml = MonkeyLearn('your api key')
my_text = """
When it comes to evaluating the performance of keyword extractors, you can use some of the standard metrics in machine learning: accuracy, precision, recall, and F1 score. However, these metrics don’t reflect partial matches; they only consider the perfect match between an extracted segment and the correct prediction for that tag.
"""
data = [my_text]
model_id = 'model id'
result = ml.extractors.extract(model_id, data)
dataDict = result.body
for item in dataDict[0]['extractions'][:10]:
	print(item['parsed_value'])