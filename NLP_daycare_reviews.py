#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 09:52:33 2019

@author: wp
"""

#################################
###WEBSCRAPING PORTION OF CODE###
#################################
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
from fake_useragent import UserAgent
import numpy as np

###URLS
url_list = ('https://www.parkslopeparents.com/515_Daycare-for-Under-1s.html', 
        'https://www.parkslopeparents.com/387_Daycare-Providers.html',
        'https://www.parkslopeparents.com/387_Daycare-Providers/Page-2.html')

url_root = 'https://www.parkslopeparents.com'

#DC_urls = pd.DataFrame()


def review_list(urllist):
    
    DC_urls = pd.DataFrame()
    
    for url in urllist:
    ## Grab content from HTML tags
        content = requests.get(url)
        content.encoding = 'utf-8'
        soup = BeautifulSoup(content.text, 'html.parser')
        divs = soup.find_all(class_='title-link')
        name = [word.string for word in divs]
        url_list2 = [url_root + a['href'] for a in divs]        
        DC_urls = DC_urls.append(pd.DataFrame({
                                        'day care name' : name, 
                                        'review url': url_list2,
                                        'root url': url,
                                        }), ignore_index = True)
        
        #add dummy variable if DC has infants
        DC_urls['hasInfants'] = DC_urls['root url'].apply(lambda x: 1 if x == url_list[0] else 0)
     
    return DC_urls
        
DC_urls = review_list(url_list)      
#rows with duplicated daycare names and has infants
#return the duplicated daycare name
DCs = DC_urls['day care name']

drop_rows = DC_urls[DCs.isin(DCs[DCs.duplicated()]) & (DC_urls['hasInfants'] == 0)]

DC_urls.drop(drop_rows.index, inplace = True)
#use 'day care name' because "Peter Pan and Wendy Daycare" has different urls to reviews, but is in both lists
#hasInfants == 1 and hasInfants == 0 
#28 daycares with duplicate names


ua = UserAgent() 



def get_reviews(list_of_review_urls):
### reviews for all daycares
    review_coms_df = pd.DataFrame()
    review_subs_df = pd.DataFrame()
    for url in list_of_review_urls.iloc[:,1]:
        reviews_for_dc = requests.get(url, {"User-Agent": ua.random})
        reviews_for_dc.encoding = 'utf-8'
        review_soup = BeautifulSoup(reviews_for_dc.text, 'html.parser')

        for link in review_soup.select('.reviewed-by-date~ div+ div'): 
	
    #reviews are mixed comments and subjects tags
            review_coms_df = review_coms_df.append(pd.DataFrame({'comments': [link.get_text()],
                                                              'review url' : url, 
                                                              }, index = [0]), ignore_index=True)
    
        for link in review_soup.select('.reviewed-by-date+ div'): ##ALL SUBJECT  
            review_subs_df = review_subs_df.append(pd.DataFrame({'subject': [link.get_text()],
                                                             'review url' : url, 
                                                             }, index = [0]), ignore_index=True)
    
    ##### get the raw reviews
    #Drop blank rows and duplicates from comment df
    review_coms_df = review_coms_df[review_coms_df.comments != ""] #
    review_coms_df.drop_duplicates()

    #Drop blank rows and duplicates from subject df
    review_subs_df = review_subs_df[review_subs_df.subject != ""] #
    review_subs_df.drop_duplicates()

    #####  select comments only from the subject df. 
    # String to be searched in start of string  
    search ="Comment:"

    # boolean series returned with False at place of NaN 
    bool_series = review_subs_df["subject"].str.startswith(search, na = False) 
  
    # displaying filtered dataframe 
    comments_in_subject = review_subs_df.copy()
    comments_in_subject = comments_in_subject[bool_series] 

    # update column name to "comments"
    comments_in_subject.rename(columns={'subject':'comments'}, inplace=True)

    #### DF of just raw comment text
    all_comments = pd.DataFrame()
    all_comments = review_coms_df.copy()
    all_comments = all_comments.append(comments_in_subject, ignore_index = True)
    all_comments.rename(columns={'comments':'raw_comments'}, inplace=True)
    #DF has 1233 - 457 reviews for infants, 776 reviews for noninfants
    #DF - 1427 - 503 reviews for infants, 924 for non infants
    return all_comments

all_comments = get_reviews(DC_urls)
##############################
###CLEANING PORTION OF CODE###
##############################

##review fragments
Review_fragments = ["Program:", "PROGRAM:", "program:", "program name:", "Length of time:", "Length of Time:", "Review date", "Review:", "REVIEW:", "Type of facility:", "Comment:", "COMMENT:", "comment:", "Type:", "About the facility", "KID AGE:", "AGE:", "Child's Age:", "Child’s Age:", "Child's age:","TIME:", "Length of time:", "Do you recommend? (Hi/Rec/DoNot)", "Do You Recommend? (Hi/Rec/DoNot)", "Do you recommend (High Rec, Rec, Rec w/ Reservations, Do Not Rec)?", "Recommend \(Hi/Rec/DoNot:", "Do you recommend? (Hi/Rec/DoNot)", "Â—What year(s) does your review cover?","Â—How old is/was your child(ren) when they attended the preschool/daycare/playgroup?","Â—Review of experience?","Â—Would you recommend?","Â—Piece of advice?"]
Review_fragments_dict = {k:"" for k in Review_fragments}
all_comments['cleaner_comments'] = all_comments['raw_comments']
all_comments['cleaner_comments'].replace(Review_fragments_dict, inplace=True, regex=True)
all_comments['cleaner_comments'].replace({"\n": " "}, inplace=True, regex=True)
print (all_comments['cleaner_comments'])

## replace daycare names and name fragments with a generic name placeholder 'DayCareNamePlaceholder'
DayCareName_Fragments = ["sunflower", "Sunflower", "SUNFLOWER", "Bright Horizons", "bright horizons", "tiny steps", "Tiny Steps", "Bumblebees", "bumblebees", "Bumblebee", "bumblebee", "Honeydew Drop", "HoneyDew Drop", "honeydew drop","Honeydew", "HoneyDew", "honeydew", "NY Kids Club", "ny kids club", "chai", "Adorable Pumpkins", "Bambi", "Aleyna's daycare", "Bumble Bee Daycare", "Bumble Bee", "Daddy's daycare", "Daddy's Daycare", "Daddy's Day Care", "Daddy's", "Daisys daycare", "Daisy Daycare", "Daisy Family Daycare", "Daisy Family Day Care", "Daisy's", "Eladia's kids", "Eladia's Kids", "Eladia's", "Etienne's Family Daycare", "Gosia's Day Care", "Gosia's daycare", "Gosia's", "Happy Baby daycare", "Happy Baby", "HOC", "Helen Owen Carey", "Hip Hippo daycare", "Hip Hippo", "Ia's", "Irving Golding", "Kiddy Citi", "Yale Youngsters", "Kids Run Around", "KRA", "Kidcare", "Le jardin de Louise","Le Jardin de Louise", "LJDL", "Little Flowers", "les Bijoux de Miley", "Les Bijoux de Miley", "Little Mushrooms", "Little Stars", "Melissa's Play School", "Melissa's", "Midwood Early Learning", "New York Kids Club", "NY Kids' Club", "NY kids club", "NYKids Club", "NY Kidsclub", "MDS", "Aardvarks", "Peter Pan & Wendy daycare","Peter Pan & Wendy Day care","Peter Pan and Wendy Daycare", "Peter Pan and Wendy", "Peter Pan & Wendy", "Pre-Preschool with Polly", "Polly's program", "Kidville", "Kids Club", "Zusin", "St. Johns Kidz", "St. John'z Kidz", "St. John'z", "SJK", "Young Ideas Day Care"]
day_care_names_list = ['Bumble Bee Daycare Inc', 'Bumblebees R Us - 8th Street, Park Slope Location', 'Cititots Infant Toddler', 'Daisy Day Care', 'DinoKing Home Daycare', "ELADIA'S KIDS", "Gosia's Daycare", 'Hanover Place Child Care Center', 'Happy Baby Daycare', 'Happy Hours Day Care ', 'HONEYDEW DROP DAYCARE', "Ia's Learning Tree", "Kids' Care on Douglass", 'Le Jardin de Louise - French green day care', 'Little Snowflakes', 'Luz Morales Family Daycare', 'Midwood Early Learning Center', 'Natalya’s', 'Peter Pan & Wendy Daycare', 'ROOTS & WINGS CHILDCARE PROGRAM', "Sofia's Daycare", 'SUNFLOWER ACADEMY CHILD CARE PARK SLOPE', 'Sunflower Playhouse (Prospect Heights)', 'Tiny Steps - 4th Ave Park Slope Location', 'Tiny Steps - Bergen Street Location', 'Tiny Steps - South Slope location', 'Zusin Family Daycare', "[CLOSED] ILENE'S SUNFLOWER CHILD CARE (WINDSOR TERRACE)", '3 Jams ', 'Adorable Pumpkins DayCare', "Aleyna's Daycare", "Ali's Stars Group Family Day ", "Alistar's Group Family Day Care/ Alistar's Play Place", 'Bambi Child Care', 'Bambi IV', 'Beansprouts', 'Bey-Bee-Sit Daycare', 'Bija Kids', 'Bkid Brooklyn ', 'Bright Horizons', 'Brooklyn Free Space', 'Brooklyn Friends School', 'Brooklyn Sandbox', 'Brooklyn Treehouse Preschool', 'Building Blocks BK', 'Building Blocks Brooklyn', 'Bumblebees R Us - Classon Ave, Prospect Heights Location', 'Bumblebees R Us - Prospect Ave, South Slope Location', 'Chai Daycare', 'Choo Choo Train Project', 'Chubby Cheeks - Ozlem Seckin', 'Cobble Hill Playschool', 'Creative Steps Early Care & Education Center - University Settlement', "Daddy's Daycare", "Daddy's Daycare 4", "Denise's Childcare", 'ELAINE RAMIREZ', 'Elemental Arts Montessori School', "Etienne's Family Day Care (Myriam Etienne)", 'Friends of Crown Heights', 'Greenwood Heights Playgroup', 'Helen Owen Carey Child Development Center', 'Hip Hippo Daycare', 'Honeydew Drop Nook', 'HoneyDew Drop Safari', 'Imagine Early Learning Center', 'Irving-Golding School', 'Ivy League Early Learning Academy ', 'Kiddi City', 'Kids Run Around Daycare', 'KinderHaus Brooklyn', 'La Petite Colline', 'Ladybug Family Daycare', "Learning and Fun Daycare at St. George's Academy", 'Les Bijoux de Miley - Miley Diarrassouba ', "Let's Play and Learn, Inc. ", 'Little Flowers Daycare', 'Little Jewels Day Care ', 'LITTLE MUSHROOMS (CLOSED)', 'Little Stars Day Care Center', "Melissa's Playschool", 'Mini Minders', 'MONTESSORI DAY SCHOOL OF BROOKLYN', 'NY Kids Club - Brooklyn Heights', 'NY Kids Club - Park Slope', 'OPEN HOUSE NURSERY SCHOOL', 'Park Avenue KinderCare', 'Park Slope Montessori ', 'Park Slope Schoolhouse', "Parker's Place", 'Peter Pan & Wendy Daycare', 'Play Outside the Lines', 'Pre-PreSchool with Polly', 'Prospect Group Family Day Care', 'Purpose Driven Learning Academy', "Regina's Daycare", "RORY'S ROOM", 'Saint Saviour Catholic Academy', 'Soles Playtime Inc', 'St. Francis Xavier Catholic Academy', "ST. JOHN'S KIDZ", "St. John's Kidz Too", "St John's Kidz", 'Tiny Scientist Day School', 'Tiny Steps - Prospect Heights/ St Marks Ave Location', 'Tiny Steps - St Johns Place ', 'Tiny Steps Daycare - Baltic Street Location', 'TLC Kids', 'TRINITY PRESCHOOL - NURSERY', 'Wortspiele', "Yoko's Daycare ", 'Young Ideas Group Family Daycare']
#day_care_names = list(DC_urls['day care name']) + list(dc_No_u1['day care name']) + DayCareName_Fragments
day_care_names = day_care_names_list + DayCareName_Fragments
v = 'DayCareNamePlaceholder'
d = {k:v for k in day_care_names}
all_comments['cleaner_comments'].replace(d, inplace=True, regex=True)
print (all_comments)

#common terms - pre K, preschool, daycare
## replace daycare names and name fragments with a generic name placeholder 'DayCareNamePlaceholder'
PreK_term_list = ["pre-K", "Pre-K", "pre-k", "Pre-k"]
d = {k:"prek" for k in PreK_term_list}
all_comments['cleaner_comments'].replace(d, inplace=True, regex=True)

PreSchool_term_list = ["pre-school", "pre-School", "Pre-School", "pre-skool", "pre-Skool", "Pre-Skool", "preskool", "preSkool", "PreSkool"]
d = {k:"preschool" for k in PreSchool_term_list}
all_comments['cleaner_comments'].replace(d, inplace=True, regex=True)

daycare_term_list = ["day care", "Day Care","dayCare", "Daycare", "DayCare", "Day-Care", "day-care"]
d = {k:"daycare" for k in daycare_term_list}
all_comments['cleaner_comments'].replace(d, inplace=True, regex=True)



####Replace possesive first name with dummy string.
## possesive nature possibly indicative of daycare name or teacher

from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict' 
wordlists = PlaintextCorpusReader(corpus_root, '.*') 
wordlists.fileids()
propernames = [w for w in wordlists.words('propernames')]
import nltk
names = nltk.corpus.names
#names.fileids() #['female.txt', 'male.txt']
male_names = names.words('male.txt')
female_names = names.words('female.txt')

propernames = [w for w in wordlists.words('propernames')] # 1312 names
female_names = [w for w in female_names] #5001 names
male_names = [w for w in male_names] #2943 names

all_names = female_names  + male_names + propernames #7960 names
all_names = set(all_names)
all_names.remove('Eve')

len(set(all_names))#7960 
all_names.remove("-")

string = "'s "
first_names_for_DCN = [x + string for x in all_names]


possessive_name_dict = {k:'PossessiveNamePlaceholder' for k in first_names_for_DCN}
all_comments['cc_names'] = all_comments['cleaner_comments']
all_comments['cc_names'].replace(possessive_name_dict, inplace=True, regex=True)


all_names_ = [x + " " for x in all_names]
name_dict = {k:'firstNamePlaceholder' for k in all_names_}
all_comments['cc_names'].replace(name_dict, inplace=True, regex=True)
all_comments['ccn_No_Num'] = all_comments['cc_names']

from nltk.corpus import stopwords
stop_words = stopwords.words('english')  

###
### function to remove puntuation and numbers 
##

def clean_up_sw(var):
    tmp = re.sub('[^a-zA-Z]+', ' ', var) #### consider adding back 0-9
    tmp = [word for word in tmp.split() if word not in stop_words]
    
    tmp = ' '.join(tmp)
    return tmp

## lambda clean up function
all_comments_no_sw = pd.DataFrame(map(lambda x: (clean_up_sw(x)), all_comments["ccn_No_Num"]))

###update column in all_comments
all_comments.ccn_No_Num = all_comments_no_sw[0]
###

import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
all_comments['ccn_No_Num_lemmas'] = all_comments.ccn_No_Num.apply(lemmatize_text)




###############################
###MODELLING PORTION OF CODE###
###############################

        
#from nltk.corpus import stopwords
#import re
#import os
#import numpy as np
#import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

##LEMMAS 
#my_vec_tfidf = TfidfVectorizer()
#my_xform_tfidf_lemma = my_vec_tfidf.fit_transform(all_comments['ccn_No_Num_lemmas'].values.astype('U')).toarray()
#col_names_lemma = my_vec_tfidf.get_feature_names()
#my_xform_tfidf_lemma = pd.DataFrame(my_xform_tfidf_lemma, columns=col_names_lemma) 


my_vec_tfidf = TfidfVectorizer()
my_xform_tfidf = my_vec_tfidf.fit_transform(all_comments['ccn_No_Num'].values.astype('U')).toarray()
col_names = my_vec_tfidf.get_feature_names()
my_xform_tfidf = pd.DataFrame(my_xform_tfidf, columns=col_names) 

#determine number of components required to achieve user specified var_target 
def iterate_var(my_xform_tfidf_in, var_target, data_slice):
    var_fig = 0.0
    cnt = 1
    while var_fig <= var_target:
        pca = PCA(n_components=cnt)
        my_dim = pca.fit_transform(my_xform_tfidf_in)
        var_fig = sum(pca.explained_variance_ratio_)   
        cnt += 1
    cnt -= 1
    print (cnt)
    pca = PCA(n_components=cnt)
    my_dim = pca.fit_transform(my_xform_tfidf_in)
    var_fig = sum(pca.explained_variance_ratio_) 
    print (var_fig)

    return my_dim, pca

#call up PCA function of above and determine optimal component count for a 'small' test size
my_dim, pca = iterate_var(my_xform_tfidf, 0.95, 200)
### #105 am
###  Results = my_dim, pca = iterate_var(my_xform_tfidf, 0.95, 200)
###  930 #NUM COMPONENETS TO GET TO EXPLAIN VARIANCE RATIO OF 95%
###  0.9502424666824185 ## THE EXPLAINED VARIANCE RATIO
###
###  approx 30 minutes to run the PCA.
###
#function for optimal parameters to set for random forest
def grid_search_func(param_grid, the_mode_in, the_vec_in, the_lab_in):
    grid_search = GridSearchCV(the_mode_in, param_grid=param_grid, cv=5)
    best_model = grid_search.fit(the_vec_in, the_lab_in)
    max_score = grid_search.best_score_
    best_params = grid_search.best_params_

    return best_model, max_score, best_params

#paramters to exhaustively iterate through
param_grid = {"max_depth": [10, 50, 100],
              "n_estimators": [16, 32, 64],
              "random_state": [1234]}

#call up model from above
clf_pca = RandomForestClassifier()
#call up grid search optimal
gridsearch_model, best, opt_params = grid_search_func(
        param_grid, clf_pca, my_xform_tfidf, all_comments.hasInfants)

#call up new model and input optimal paramters from above
clf_pca = RandomForestClassifier()
clf_pca.set_params(**gridsearch_model.best_params_)
clf_pca.fit(my_dim, all_comments.hasInfants)

###
###
###
#RANDOM FOREST OUTPUT
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                       max_depth=50, max_features='auto', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=64,
#                       n_jobs=None, oob_score=False, random_state=1234,
#                       verbose=0, warm_start=False)
####

clf = RandomForestClassifier(n_estimators=64, max_depth=2, random_state=100)
clf.fit(my_xform_tfidf, all_comments.hasInfants)  


pca = PCA(n_components=500) 
print (sum(pca.explained_variance_ratio_)) 
##100: 0.3300237300924692
##400: 0.6827017204824363
##500: 0.7560880398724122
##600: 0.8165942262823354
##700: 0.8665780884880834



#split the data into training data and testing data.
from sklearn.model_selection import train_test_split# implementing train-test-split
X_train, X_test, y_train, y_test = train_test_split(my_dim, all_comments.hasInfants, test_size=0.20, random_state=66) #20 


# random forest model creation
from sklearn import model_selection
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)# predictions
rfc_predict = rfc.predict(X_test)
rfc_predict = np.array(rfc_predict)
y_test = np.array(y_test)
y_pred = rfc_predict
#rfc_predict = rfc.predict(X_test)
#accuracy_score(y_true, y_pred)

precision_recall_fscore_support(y_test, y_pred, average=None)

## RF - 500 - score table. 
#(array([0.64676617, 0.5952381 ]), #Precision score
# array([0.88435374, 0.26041667]), #recall score
# array([0.74712644, 0.36231884]), #F1 score
# array([147,  96])) #The number of occurrences of each label in y_true

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) 
#0.5967078189300411 #with 700 components
#0.6172839506172839 #with 500 components - highest accuracy.  0.6378600823045267
#0.0.588477366255144 #with 400 components


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("Precision Score:")
precision_score(y_test, y_pred, average=None)

print("Recall Score:")
recall_score(y_test, y_pred, average=None)

print("F1 Score:")
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average=None)


from sklearn.metrics import confusion_matrix
##confusion matrrix
# Create random forest classifier instance
trained_model = rfc.fit(X_train,y_train)
predictions = trained_model.predict(X_test)
 
print ("Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train)))
print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
print (" Confusion matrix ")
print(confusion_matrix(y_test, predictions))
#20% test 
#RF results
#Train Accuracy ::  0.9814624098867147
#Test Accuracy  ::  0.6460905349794238
# Confusion matrix  
# [[138   9]
# [ 77  19]]

# 30% test
#Train Accuracy ::  0.9870435806831567
#Test Accuracy  ::  0.6356164383561644
# Confusion matrix  
# [[206  23]
# [110  26]]

#40%
#Train Accuracy ::  0.9862637362637363
#Test Accuracy  ::  0.6234567901234568
# Confusion matrix  
# [[270  31]
# [152  33]]

#25%
#Train Accuracy ::  0.9769230769230769
#Test Accuracy  ::  0.6151315789473685
# Confusion matrix  
# [[165  24]
# [ 93  22]]



######################################
###COMPARING MODELS PORTION OF CODE###
######################################

#from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
naiv_y = np.zeros((243))

X, y = my_xform_tfidf, all_comments.hasInfants

clf1 = LogisticRegression(random_state=1)


clf2 = RandomForestClassifier(n_estimators=64, max_depth=2, random_state=100)
#clf2.fit(my_xform_tfidf, all_comments.hasInfants) 
clf3 = MultinomialNB()

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('mnb', clf3)],
    voting='hard')

for clf, label in zip([clf1, rfc.fit, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#Accuracy: 0.64 (+/- 0.07) [Logistic Regression]
#Accuracy: 0.62 (+/- 0.00) [Random Forest]
#Accuracy: 0.62 (+/- 0.00) [naive Bayes]
#Accuracy: 0.62 (+/- 0.00) [Ensemble]

#naive test - assign all values the most common classification ('hasInfants' = 0)
print(accuracy_score(y_test, naiv_y))
#Accuracy: 0.60  [All_no_infants]

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='f1', cv=5)
    print("F1: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#F1: 0.36 (+/- 0.08) [Logistic Regression]
#F1: 0.00 (+/- 0.00) [Random Forest]
#F1: 0.01 (+/- 0.02) [naive Bayes]
#F1: 0.01 (+/- 0.02) [Ensemble]

print(f1_score(y_test, naiv_y))
#F1: 0.0  [All_no_infants]
    
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='precision', cv=5)
    print("Precision: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

#Precision: 0.60 (+/- 0.17) [Logistic Regression]
#Precision: 0.00 (+/- 0.00) [Random Forest]
#Precision: 0.07 (+/- 0.13) [naive Bayes]
#Precision: 0.07 (+/- 0.13) [Ensemble]

print(precision_score(y_test, naiv_y))
#Precision: 0.0  [All_no_infants]    
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, scoring='recall', cv=5)
    print("Recall: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


#Recall: 0.26 (+/- 0.05) [Logistic Regression]
#Recall: 0.00 (+/- 0.00) [Random Forest]
#Recall: 0.01 (+/- 0.01) [naive Bayes]
#Recall: 0.01 (+/- 0.01) [Ensemble]
print(recall_score(y_test, naiv_y))
#Recall: 0.0 [All_no_infants]

precision_recall_fscore_support(y_test, naiv_y, average=None)

print(confusion_matrix(y_test,naiv_y))
print(classification_report(y_test,naiv_y))
print(accuracy_score(y_test, naiv_y))



#############################
###      END OF CODE      ###
#############################
