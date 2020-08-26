from rake_nltk import Rake
r = Rake()

my_text = """
When it comes to evaluating the performance of keyword extractors, you can use some of the standard metrics in machine learning: accuracy, precision, recall, and F1 score. However, these metrics don’t reflect partial matches; they only consider the perfect match between an extracted segment and the correct prediction for that tag.

Fortunately, there are some other metrics capable of capturing partial matches. An example of this is ROUGE.
"""
r.extract_keywords_from_text(my_text)
keywordList 					= []
rankedList 						= r.get_ranked_phrases_with_scores()
for keyword in rankedList:
	keyword_updated 			= keyword[1].split()
	keyword_updated_string 		= " ".join(keyword_updated[:2])
	keywordList.append(keyword_updated_string)
	if(len(keywordList)>9):
		break


print(keywordList)