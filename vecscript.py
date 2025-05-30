import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('punkt')

from nltk.corpus import wordnet
import gensim.downloader as api
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

"""
Author: Kara Leier
Date: May 8, 2025
Purpose: Honour's Project, Jan-Apr 2025.

This is the start of the code that is dedicated to creating vectors of the nouns in the dataset. 

The data set is extracted from the Elford and Elford Denesuline dictionary.
This code is extremely based on the organization of the spreadsheet that was created by Dr. Olga Lovick and PhD candidate Olga Kriukova.

Each entry in the dictionary takes up one row in the spreadsheet.
Each entry is turned into two vectors to obtain two sets of vectors.

The first set, which I will label, is made entirely from computational methods using the word2vec model.

The second set, which I will label, is made from our manual classifications. THis is the "wordnet' column in the spreadsheet.
It is done by using the definition from wordnet of the selected wordnet words and creating a vector from that.
"""

# two separate models for the two sets of vectors
word2vec_model = api.load('word2vec-google-news-300')
eng_wv = api.load('word2vec-google-news-300')

# Here I am bringing in only the 4 column that I want
data = pd.read_csv('nouns.csv', encoding='utf-16le', names=['English', 'POS', 'Wordnet', 'Dene'])

# Clean the data here
data = data.dropna(axis=0)
trim_data = data[["English","Wordnet", "Dene"]]
trim_data['en_len'] = trim_data['English'].apply(lambda x: len(x))
trim_data['wn_len'] = trim_data['Wordnet'].apply(lambda x: len(x))
en_data =  trim_data[trim_data.en_len > 1]
en_data = trim_data[trim_data.wn_len > 1]

# Create new objects to hold the data as I process it
new_data = {'English': [], 'Dene': []}
wn_data = {'Wordnet': [], 'Dene':[]}
vecs = []

# Get the vectors of the English glosses
# This is the FIRST set
for index, row in en_data.iterrows():
	new_data["English"].append(row['English'].strip().replace('(', '').replace(')', '').replace(',', '').replace("see also ", '').replace('"', '').split(" "))
	new_data["Dene"].append(row["Dene"])
	wn_data["Wordnet"].append(row["Wordnet"].strip().replace(" (", "$").replace("(", "").replace(",", '').split("$"))
	wn_data["Dene"].append(row["Dene"])

clean_data = pd.DataFrame(data=new_data, index=[x for x in range(0, len(en_data))])
clean_wn_data = pd.DataFrame(data=wn_data, index=[x for x in range(0, len(en_data))])

for index, row in clean_data.iterrows():
	try:
		avrg = 0
		main = 0
		i = 0
		valid = [token for token in row["English"] if token in eng_wv]
		if len(valid) == 0:
			continue
		for each in valid:
			if i == 0:
				main+=eng_wv[each]
			else:
				avrg+=eng_wv[each]
			i+=1
		if len(valid) == 1:
			fin_avrg = main
		else:
			avrg = avrg/(len(valid)-1)
			fin_avrg = (main*0.4)+(avrg*0.6)
		vecs.append([fin_avrg, row["Dene"], row["English"]])
	except KeyError as e:
		pass

# Get the Wordnet vectors
# This is the SECOND set
wn_vecs = []
for index, row in clean_wn_data.iterrows():
	try:
		avrg = 0
		for each in row["Wordnet"]:
			pos = ""
			stem = ""
			sense = ""
			posi = 0
			for letter in each:
				if letter == "(":
					pass
				if letter != ")" and posi == 0:
					pos += letter
				elif letter == ")" and posi == 0:
					posi = 1
				elif letter != "#" and posi == 1:
					stem+=letter
				elif letter == "#" and posi == 1:
					posi = 2
				else:
					sense += letter
			wn_sample = stem+"."+pos+"."+"0"+sense
			wn_def = wordnet.synset(wn_sample).definition()
			tokens = nltk.word_tokenize(wn_def.lower())
			valid_tokens = [token for token in tokens if token in word2vec_model]
			vectors = [word2vec_model[token] for token in valid_tokens]
			main_word = stem.strip().replace("_", " ").replace("-", " ").split()
			main_vec = [word2vec_model[word] for word in main_word]
			main_vecs = np.mean(main_vec, axis=0).reshape(1, -1)
			avg_vector = np.mean(vectors, axis=0).reshape(1, -1)
			fin_vec = (main_vecs*0.4)+(avg_vector*0.6)
			wn_vecs.append([fin_vec, row["Dene"], row["Wordnet"]])
	except Exception as e:
		pass

# Combine the two sets here back to one.
wn_final = []
vec_final = []
for each in wn_vecs:
	for item in vecs:
		if (each[1] == item[1]):
			wn_final.append(each)
			vec_final.append(item)
			break

# Finally, calculate and record the cosine similarity
vec_sim_matrix = []
for i in range(len(wn_final)):
	#item = vec english, dene, wn english, similarity, man vec, comp vec
	item = []
	# comparing comp-based vector to manual-based vector
	similarity = cosine_similarity([vec_final[i][0]], wn_final[i][0])
	item.append(vec_final[i][2])
	item.append(vec_final[i][1])
	item.append(wn_final[i][2])
	item.append(similarity)
	vec_sim_matrix.append(item)

final = pd.DataFrame(vec_sim_matrix, columns=["English Gloss", "Dene", "Wordnet", "Similarity"])
final.to_csv("nouns_out.csv", encoding='utf-8-sig', header=True)

'''
This is the code to create the graphs I used for my presentation but is not necessary for the vectors.
--
y = final["Similarity"]

y_data = []
for each in y:
	y_data.append(each[0][0])

y_data = pd.DataFrame(y_data)
print("Nouns\n", y_data.describe())

final = final.sort_values(by=["Similarity"])
print(final.head())
print(final.tail())

sns.histplot(y_data, bins=50, kde=True)
mp.title("Noun Similarities")
mp.xlabel("Cosine Similarity")
mp.ylabel("Frequency")
mp.grid(True)
mp.savefig("nouns.png")
'''

