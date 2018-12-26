save.image()
rm = remove(list = ls())
## set Working Directory
setwd("C:/Users/palak/OneDrive/Desktop/Data Mining/Assignment 4")

### install and load packages

# install.packages('tm')
# install.packages('tidytext')
# install.packages("SnowballC") # for text stemming
# install.packages("wordcloud") # word-cloud generator 
# install.packages("RColorBrewer") # color palettes
# install.packages('tokenizers')
# install.packages('quanteda')
# install.packages('wordnet')
# install.packages('slam')
# install.packages('sqldf')
# install.packages('naivebayes')

library("tm")
library('quanteda')
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library('ggplot2')
library('dplyr') 
library('tidytext') # for word count
library('tokenizers') # not yet used
library('Matrix') # for sparse matrix
library('wordspace') # to normalize rows
library('slam')  # column sum for positive and negative words - Part B
library('sqldf') # afin dictionary operation 
library('e1071') # for naive bayes
library('glmnet') # for lasso regression
library('naivebayes')
library('randomForest')
# library('wordnet') # https://github.com/cran/wordnet/blob/master/R/dictionary.R -- COde to initialise wordnet dictionary

## load data
yelp_reviews <- read.csv("yelpRestaurantReviewSample_50K.csv",header = TRUE, sep = ";", stringsAsFactors = TRUE)
## Convert reviews to text
yelp_reviews$text <- as.character(yelp_reviews$text)

reviews <- quanteda::corpus(yelp_reviews$text)
class(reviews)

## ------------------------------ Data Treatment ------------------------------ ## 

### Tokenize, remove numbers, punctutation symbols, hyphens
reviews_Tokens <- quanteda::tokens(reviews, remove_numbers = TRUE,  remove_punct = TRUE, remove_symbols = TRUE, remove_hyphens = TRUE)

## convert to Document term matrix, convert to lower case and remove stopwords
dtm_full <- dfm(x = reviews_Tokens, tolower = TRUE, stem = FALSE, remove = stopwords('english') )

## use weights for each words a prop to total words in documents(term frequency)
dtm_full_tf <- dfm_weight(x = dtm_full, scheme = 'prop')

## ====================================== Part B ====================================== ##
## ------------------------------ Identifying Positive and Negative Words ------------------------------ ##
## filter for document freq
dtm_full_termfreq <- dfm_weight(dfm_select(x = dfm_trim(x = dtm_full, min_termfreq = 0.00005, termfreq_type = "prop" ), min_nchar = 3, max_nchar = 20),scheme = 'prop')

dim(dtm_full_termfreq)
# View(dtm_full_termfreq[1,1:10])

dtm_stars <- dtm_full_termfreq*yelp_reviews$stars
word_stars_total <- slam::col_sums(dtm_stars, na.rm = T) # https://stackoverflow.com/questions/21921422/row-sum-for-large-term-document-matrix-simple-triplet-matrix-tm-package
word_total <- slam::col_sums(dtm_full_termfreq, na.rm = T) 
length(word_total)
final_score <- word_stars_total/word_total
## QC

length(final_score)
# View(dtm_stars[1,1:10])
# View(yelp_reviews$stars[1:10])
# View(word_stars_total[1:10])

positive_words <- sort(final_score, decreasing = TRUE)[1:500]
negative_words <- sort(final_score, decreasing = FALSE)[1:500]
View(positive_words)
View(negative_words)

#-------- Word Cloud --------#
d <- data.frame(names(positive_words), round((positive_words^3)*1000,0))
colnames(d) <- c('word', 'freq')
View(d[1:20,])
set.seed(100)
wordcloud (words = d$word, freq = d$freq, min.freq = 3,
           random.order = F, max.words = 200, scale = c(1,5), random.color = TRUE, colors=brewer.pal(8, "Dark2"), rot.per=0.35)


d1 <- data.frame(names(negative_words), round((1/negative_words)*100,0))
colnames(d1) <- c('word', 'freq')
str(d1)

set.seed(100)
wordcloud (words = d1$word, freq = d1$freq, min.freq = 3, random.order = F, max.words = 200, scale = c(1,7), 
           random.color = TRUE, colors=brewer.pal(8, "Dark2"), rot.per=0.35)


p <- ggplot(subset(d, freq>90000), aes(x = reorder(word, -freq), y = freq)) +
  geom_bar(stat = "identity") + 
  theme(axis.text.x=element_text(angle=45, hjust=1))
p

## ====================================== Part C ====================================== ##
## filter for words length
dtm_full_wordlength <- dfm_select(x = dtm_full_tf, min_nchar = 3, max_nchar = 20)
dtm_full_wordlength

View(dtm_full_wordlength[1,1:10])
dim(dtm_full_wordlength)

## ------------------------------ Bing Liu Dictionary ------------------------------ ##
#folder with positive dictionary
pos <- scan('C:/Users/palak/OneDrive/Desktop/Data Mining/Assignment 4/positive words.txt', what='character', comment.char=';')


#folder with negative dictionary
neg <- scan('C:/Users/palak/OneDrive/Desktop/Data Mining/Assignment 4/negative words.txt', what='character', comment.char=';')

## ------------------------------ Sentiment Analysis using Positive and Negative List - Bing Liu ------------------------------ ##

### Positive words
dtm_full_pos <- dfm_select(x = dtm_full_wordlength, pattern = pos)
dtm_full_neg <- dfm_select(x = dtm_full_wordlength, pattern = neg)
dim(dtm_full_wordlength)
dim(dtm_full_pos) #[1] 50000  1359
dim(dtm_full_neg) #[1] 50000  2506

positiveSum <- apply(dtm_full_pos, MARGIN = 1, sum)
negativeSum <- apply(dtm_full_neg, MARGIN = 1, sum)

prediction <- ifelse(positiveSum > negativeSum , 1, 0)
actual <- ifelse( as.numeric(yelp_reviews$stars) <= 2, 0, 
                  ifelse (as.numeric(yelp_reviews$stars) >=4 , 1, "neutral"))                  
table(actual)
table(yelp_reviews$stars)

star_prediction <- as.data.frame(cbind(positiveSum, negativeSum, as.numeric(yelp_reviews$stars), prediction, actual))
colnames(star_prediction) <- c("positiveSum","negativeSum", "stars","prediction", "actual")
head(star_prediction)

bing_liu_result <- table(pred = star_prediction$prediction, true = star_prediction$actual)

## ------------------------------ Load Harvard Dictionary ------------------------------ ##
#folder with positive dictionary
harv_pos <- scan('C:/Users/palak/OneDrive/Desktop/Data Mining/Assignment 4/Harvard_positive_words.txt', what='character', comment.char=';')


#folder with negative dictionary
harv_neg <- scan('C:/Users/palak/OneDrive/Desktop/Data Mining/Assignment 4/Harvard_negative_words.txt', what='character', comment.char=';')

## ------------------------------ Sentiment Analysis using Positive and Negative List - Haravrd  ------------------------------ ##

### Positive words
dtm_full_pos <- dfm_select(x = dtm_full_wordlength, pattern = harv_pos)
dtm_full_neg <- dfm_select(x = dtm_full_wordlength, pattern = harv_neg)

dim(dtm_full_pos) #[1] 50000  1199
dim(dtm_full_neg) #[1] 50000  1300

positiveSum <- apply(dtm_full_pos, MARGIN = 1, sum)
negativeSum <- apply(dtm_full_neg, MARGIN = 1, sum)

prediction <- ifelse(positiveSum > negativeSum , 1, 0)
actual <- ifelse( as.numeric(yelp_reviews$stars) <= 2, 0, 
                  ifelse (as.numeric(yelp_reviews$stars) >=4 , 1, "neutral"))                  
table(actual)
table(yelp_reviews$stars)

star_prediction <- as.data.frame(cbind(positiveSum, negativeSum, as.numeric(yelp_reviews$stars), prediction, actual))
colnames(star_prediction) <- c("positiveSum","negativeSum", "stars","prediction", "actual")
head(star_prediction)

harvard_result <- table(pred = star_prediction$prediction, true = star_prediction$actual)


## ------------------------------ Afinn Dictionary  ------------------------------ ##
afinn <- subset(sentiments, lexicon == "AFINN")
afinn[["sentiment"]] <-
  with(afinn,
       sentiment <- ifelse(score < 0, "negative",
                           ifelse(score > 0, "positive", "netural"))
  )
with(afinn, table(score, sentiment))
as.dictionary(afinn)


afinn_dict <- as.data.frame(afinn %>% select(c(word,score)))
length(afinn_dict$score)
table(afinn_dict$score)

## ------------------------------ Sentiment Analysis using Positive and Negative List - AFINN ------------------------------ ##

### Positive words
dtm_full_afin <- dfm_select(x = dtm_full_wordlength, pattern = afinn_dict$word)
dim(dtm_full_afin)


## subset afinn dictionary
cnames_afin <- as.data.frame (colnames(dtm_full_afin)) ## get words in dtm from the afinn dictionary
colnames(cnames_afin) <- c('word')

index <- seq(from = 1, to = 1765, by = 1)
word_index <- as.data.frame(cbind(cnames_afin, index))

result <- merge(word_index, afinn_dict, by='word')
result <- result[order(result$index),]
score <- as.matrix(result$score)
# rownames(score) <- as.vector(word_index$word)

final <- dtm_full_afin %*% score

## QC
View(final[1:100,])
View(prediction[1:100])     

prediction_afin <- as.vector(ifelse(as.vector(final) > 0 , 1, 0))
View(prediction_afin[1:100])

actual <- ifelse( as.numeric(yelp_reviews$stars) <= 2, 0, ifelse (as.numeric(yelp_reviews$stars) >=4 , 1, "neutral"))                  

afin_result <- table(pred = prediction_afin, true = actual)

## ---------------- Common words in three dictionaries ---------------- #
## positive words
df_bing <- data.frame(word = pos)
df_harv <- data.frame(word = tolower(harv_pos))
df_afin <- data.frame(word = afinn_dict$word[afinn_dict$score > 0])

df_bing_neg <- data.frame(word = neg)
df_harv_neg <- data.frame(word = tolower(harv_neg))
df_afin_neg <- data.frame(word = afinn_dict$word[afinn_dict$score < 0])

com_word_1 <- merge(df_bing, df_harv, by='word')
com_word_1_neg <- merge(df_bing_neg, df_harv_neg, by='word')
dim(com_word_1) ## 824
dim(com_word_1_neg) ## 1556

com_word_2 <- merge(df_bing, df_afin, by='word')
com_word_2_neg <- merge(df_bing_neg, df_afin_neg, by='word')
dim(com_word_2) ## 433
dim(com_word_2_neg) ## 865

com_word_3 <- merge( df_afin,df_harv, by='word')
com_word_3_neg <- merge(df_harv_neg, df_afin_neg, by='word')
dim(com_word_3) ## 349
dim(com_word_3_neg) ## 511  



## ====================================== Part D ====================================== ##

dtm_full_tfidf <- dfm_tfidf(dfm_select(x = dtm_full,min_nchar = 3, max_nchar = 20), scheme_tf = 'prop', scheme_df = 'inverse')

## ====================================== Part D (i) ====================================== ##
## ====================================== Model for Sentiment Dictionary Terms  ====================================== ##


dtm_tfidf_pos <- dfm_select(dtm_full_tfidf, pattern = pos)
dtm_tfidf_neg <- dfm_select(dtm_full_tfidf, pattern = neg)
dim(dtm_tfidf_pos) #[1] 50000  1359
dim(dtm_tfidf_neg) #[1] 50000  2506

## combine dtm from positive and negative sentiment words
dtm_tfidf_vf <- cbind(dtm_tfidf_pos,dtm_tfidf_neg)
dim(dtm_tfidf_vf) #[1] 50000  3865

## subset for star >=4 ans star <=2 
flag <- yelp_reviews$stars != 3
model_data_vf <- dfm_subset(dtm_tfidf_vf, flag) ## flag is boolean vector used to subset dtm
dim(model_data_vf) #42166  3865


actual_star <- yelp_reviews$stars[yelp_reviews$stars <= 2 | yelp_reviews$stars >= 4] ##42166 rows
actual_lable <- ifelse(actual_star >2, 1, 0)


### random sampling
set.seed(12345)
flag_sample <- sample(x = c(1, 0),  42166, replace = TRUE, prob = c(0.25, 0.75))
flag_sample <- flag_sample == 1 # for train
flag_sample_v1 <- flag_sample == 0 # for test

## Training data
model_data_train <- dfm_subset(model_data_vf, flag_sample)  
train_label <- as.factor(actual_lable[flag_sample == TRUE])
dim(model_data_train) #[1] 10395  3865

## Test Data
model_data_test  <- dfm_subset(model_data_vf, flag_sample_v1)
test_label <- as.factor(actual_lable[flag_sample == FALSE])
dim(model_data_test)

## ====================================== Naive Bayes  ====================================== ##
set.seed(12345)
NV_model1 <- naiveBayes(x = as.data.frame(model_data_train), y = train_label, prior = NULL, laplace = 1, usekernel = FALSE)
dim(model_data_train)
NV_predict_train <- predict(NV_model1,model_data_train)
NV_predict_test <- predict(NV_model1,model_data_test)

table(pred = NV_predict_train, true = train_label)
table(pred = NV_predict_test, true = test_label)

## ====================================== RANDOM FOREST  ====================================== ##

# Find the best lambda using cross-validation
set.seed(12345) 
rf <- randomForest(x = model_data_train, y = train_label, ntree = 100)


## ====================================== SVM  ====================================== ##

cost <- c(0.125, 0.25, 0.5, 1, 2,	4, 8, 16, 32, 64, 128, 256, 512)
tune_cost=tune(svm ,train.x = model_data_train, train.y = as.factor(train_label), type = 'C-classification', kernel ="linear",
               ranges =list(cost=c(0.125,	0.25,	0.5,	1,	2,	4,	8,	16,	32,	64,	128,	256,	512) ))

summary(tune_cost)

bestmod =tune_cost$best.model
summary (bestmod )

set.seed(12345)
svmfit =svm(x = model_data_train, y = train_label, scale = FALSE, type ='C-classification',kernel = 'linear', cost = 128)
y_pred_train = predict (svmfit ,model_data_train)
table(predict =y_pred_train , truth= train_label )

y_pred_test = predict (svmfit ,model_data_test)
table(predict =y_pred_test , truth= test_label)

## ====================================== Combining Dictionaries  ====================================== ##

afin_pos <- afinn_dict$word[afinn_dict$score > 0 ]
afin_neg <- afinn_dict$word[afinn_dict$score < 0 ]

positive_all <- base::union(base::union(pos, harv_pos), afin_pos)
length(positive_all)

negative_all <- base::union(base::union(neg, harv_neg), afin_neg)
length(negative_all)

## ====================================== Model for Sentiment Dictionary Terms- Using all terms ====================================== ##


dtm_tfidf_pos_all <- dfm_select(dtm_full_tfidf, pattern = positive_all)
dtm_tfidf_neg_all <- dfm_select(dtm_full_tfidf, pattern = negative_all)
dim(dtm_tfidf_pos_all) #[1] 50000  2176
dim(dtm_tfidf_neg_all) #[1] 50000  3164

## combine dtm from positive and negative sentiment words
dtm_tfidf_vf_all <- cbind(dtm_tfidf_pos_all,dtm_tfidf_neg_all)
dim(dtm_tfidf_vf_all) #[1] 50000  5340

## subset for star >=4 ans star <=2 
flag <- yelp_reviews$stars != 3
model_data_vf_all <- dfm_subset(dtm_tfidf_vf_all, flag) ## flag is boolean vector used to subset dtm
dim(model_data_vf_all) #42166  5340

actual_star <- yelp_reviews$stars[yelp_reviews$stars <= 2 | yelp_reviews$stars >= 4] ##42166 rows
actual_lable <- ifelse(actual_star >2, 1, 0)


### random sampling
set.seed(12345)
flag_sample <- sample(x = c(1, 0),  42166, replace = TRUE, prob = c(0.25, 0.75))
flag_sample <- flag_sample == 1 # for train
flag_sample_v1 <- flag_sample == 0 # for test

## Training data
model_data_train_all <- dfm_subset(model_data_vf_all, flag_sample)  
train_label <- as.factor(actual_lable[flag_sample == TRUE])
dim(model_data_train_all) #[1] 10395  5340

## Test Data
model_data_test_all  <- dfm_subset(model_data_vf_all, flag_sample_v1)
test_label <- as.factor(actual_lable[flag_sample == FALSE])
dim(model_data_test_all) #[1] 31771  5340

## ====================================== Naive Bayes - All dictionaries ====================================== ##
set.seed(12345)
NV_model1_all <- naiveBayes(x = as.data.frame(model_data_train_all), y = train_label, prior = NULL, laplace = 1, usekernel = FALSE)

NV_predict_train_all <- predict(NV_model1_all,model_data_train_all)
NV_predict_test_all <- predict(NV_model1_all,model_data_test_all)



table(pred = NV_predict_train_all, true = train_label)
table(pred = NV_predict_test_all, true = test_label)

## ====================================== RANDOM FOREST All dictionaries  ====================================== ##
set.seed(12345) 
rf_all <- randomForest(x = as.matrix(model_data_train_all), y = train_label, ntree = 100)

rf_train_pred <- predict(rf_all, model_data_train_all)
table(predict = rf_train_pred , truth= train_label ) 

dim(model_data_test_all)
rf_test_pred_1 <- predict(rf_all, model_data_test_all[1:10000,])
rf_test_pred_2 <- predict(rf_all, model_data_test_all[10001:20000,])
rf_test_pred_3 <- predict(rf_all, model_data_test_all[20001:nrow(model_data_test_all),])

t1<-table(predict = rf_test_pred_1 , truth= test_label[1:10000] ) 
t2<-table(predict = rf_test_pred_2 , truth= test_label[10001:20000] ) 
t3<-table(predict = rf_test_pred_3 , truth= test_label[20001:nrow(model_data_test_all)] ) 

t1+t3+t3


## ====================================== SVM - All dictionaries ====================================== ##

cost <- c(0.125, 0.25, 0.5, 1, 2,	4, 8, 16, 32, 64, 128, 256, 512) 
tune_cost_all=tune(svm ,train.x = model_data_train_all, train.y = as.factor(train_label), type = 'C-classification', kernel ="linear",
              ranges =list(cost=c(0.125,	0.25,	0.5,	1,	2,	4,	8,	16,	32,	64,	128,	256,	512) ))

summary(tune_cost_all) 

bestmod_all =tune_cost_all$best.model 
summary (bestmod_all)

set.seed(12345) 
svmfit_all =svm(x = model_data_train_all, y = train_label, scale = FALSE, type ='C-classification',kernel = 'linear', cost = 64) 
y_pred_train_all = predict (svmfit_all ,model_data_train_all) 
table(predict =y_pred_train_all , truth= train_label ) 

y_pred_test_all = predict (svmfit_all ,model_data_test_all) 
table(predict_all =y_pred_test_all , truth= test_label)


## ================================= Model for broader set of terms ================================= ##
## ================================= Data Preperation - Data cleaning and lemmatization  ================================= ##

###############Lemmatization###################
install.packages('textstem')
library(textstem)
## load data
yelp_reviews <- read.csv("yelpRestaurantReviewSample_50K(1).csv",header = TRUE, sep = ";", stringsAsFactors = TRUE)
## Convert reviews to text
yelp_reviews$text <- as.character(yelp_reviews$text)
is.data.frame(yelp_reviews$text)
text<-as.vector(yelp_reviews$text)




text_lemma <- lemmatize_strings(text)

text[1]
text_lemma[1]

reviews_lemma <- quanteda::corpus(text_lemma)
class(reviews)

## ------------------------------ Data Treatment ------------------------------ ##

### Tokenize, remove numbers, punctutation symbols, hyphens
reviews_Tokens_lemma <- quanteda::tokens(reviews_lemma, remove_numbers = TRUE,  remove_punct = TRUE, remove_symbols = TRUE, remove_hyphens = TRUE)

## convert to Document term matrix, convert to lower case and remove stopwords
dtm_full_lemma <- dfm(x = reviews_Tokens_lemma, tolower = TRUE, stem = FALSE, remove = stopwords('english') )

dtm_full_tfidf_lemma <- dfm_tfidf(dfm_select(x = dfm_trim(x = dtm_full_lemma, min_termfreq = 0.00001, termfreq_type = "prop" ),min_nchar = 3, max_nchar = 20), scheme_tf = 'prop', scheme_df = 'inverse')


flag <- yelp_reviews$stars != 3
model_data_vf_lemma <- dfm_subset(dtm_full_tfidf_lemma, flag) ## flag is boolean vector used to subset dtm
dim(model_data_vf_lemma) #42166  3865


actual_star <- yelp_reviews$stars[yelp_reviews$stars <= 2 | yelp_reviews$stars >= 4] ##42166 rows
length(actual_lable)
actual_lable <- ifelse(actual_star >2, 1, 0)

### add star value to dtm
model_data <- cbind(actual_lable, model_data_vf_lemma)
# update column name for star rating
colnames(model_data)[1] <- 'star'


### random sampling

########## 
flag_sample <- sample(x = c(1, 0),  42166, replace = TRUE, prob = c(0.25, 0.75))
flag_sample <- flag_sample == 1


model_data_train <- dfm_subset(model_data_vf_lemma, flag_sample)  
train_label <- as.factor(actual_lable[flag_sample == TRUE])
test_label <- as.factor(actual_lable[flag_sample == FALSE])
dim(model_data_train) #[1] 10395  3865
length(train_label)
length(test_label)
model_data_test  <- dfm_subset(model_data_vf_lemma, flag_sample==FALSE)  


NV_model1 <- naiveBayes(x = as.data.frame(model_data_train), y = train_label, prior = NULL, laplace = 1, usekernel = FALSE)
dim(model_data_train)
NV_predict_test <- predict(NV_model1,model_data_test)
table(pred = NV_predict_train, true = train_label)

##################SVM####################

cost <- c(0.125, 0.25, 0.5, 1, 2,	4, 8, 16, 32, 64, 128, 256, 512)
tune_cost=tune(svm ,train.x = model_data_train, train.y = as.factor(train_label), type = 'C-classification', kernel ="linear",
               ranges =list(cost=c(0.125,	0.25,	0.5,	1,	2,	4,	8,	16,	32,	64,	128,	256,	512) ))

summary(tune_cost)

bestmod =tune_cost$best.model
summary (bestmod )

set.seed(12345)
svmfit =svm(x = model_data_train, y = train_label, scale = FALSE, type ='C-classification',kernel = 'linear', cost = 128)
y_pred_train = predict (svmfit ,model_data_train)
table(predict =y_pred_train , truth= train_label )

y_pred_test = predict (svmfit ,model_data_test)
table(predict =y_pred_test , truth= test_label)
length(y_pred_test)
length(test_label)

################Random forest##################
install.packages("randomForest")
library('randomForest')
rf_model <-  randomForest(x = as.matrix(model_data_train),y = train_label, ntree=100, importance = T)
y_pred_train = predict (rf_model ,model_data_train,type = "class")
table(predict =y_pred_train , truth= train_label )

y_pred_test = predict (rf_model ,model_data_test)
table(predict =y_pred_test , truth= test_label)
length(y_pred_train)
length(test_label)
