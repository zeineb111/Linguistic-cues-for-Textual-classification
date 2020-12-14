
## Introduction

Under the scope of EPFL's ADA class project, we will perform a creative extension on the paper Linguistic  Harbingers  of  Betrayal. In this project we will work on the subject of Natural Language Processing (NLP) and more specially on Linguistic cues. As we will present to you in the next section, the authors of the paper worked on the messages of the diplomacy game and extracted some features to do some analysis and then train a simple model to perform a complex task, even for humans, which is detecting betrayal. To extend this work we will first try to train more complex and adapted models for this kind of tasks, like RNNs (see section 5) to see how they perform and draw some conclusions from the results. Then we will extend this work to another and completely different dataset, a dataset of news. The goal is to try and see if the same features can be used for various NLP examples.

## Related Work: The Linguistic Harbingers of Betrayal paper 

Since our work consists on a creative extension for the Linguistic Harbingers of Betrayal paper, we will describe briefly the work done by the authors to put you in the contest.   
The authors worked on the diplomacy game, which is a war-themed strategy game where friendships and betrayals are orchestrated primarily through language. They collected a dataset, that contains 500 games half of them for games that end with an attack and the other half for games without attacks. Each game contains multiple seasons, and in each season the players communicate via messages then perform and action simultaneously. The two sets are matched to get the most accurate results.   They consider an attacker the first player that breaks the friendship. For games without an attack, the attacker is chosen randomly from the two players.   They performed some preprocessing to extract the following features from the messages: sentiments (negative, positive, neutral), politeness, talkativeness, discourse markers (planning, comparison, expansion, contingency, subjectivity, premises, claims). After extracting these features, they generated some plots to see and compare the behaviours of those between the two types of players. After some analysis, they fed these features to a Logistic regression model that achieved an accuracy of 57% at detecting betrayals.   They concluded that the classifier is able
to exploit subtle linguistic signals that surface in the conversation. They  also analysed how these features evolve as we get closer to betrayal to detect imbalances and check how effective are the features at detecting long-term betrayals.

## Data collection 
As we described in the Introduction, our project consists on two part each of them working on a diffrent dataset. 

### Diplomcay dataset
The first dataset is the Diplomacy game dataset that was provided with the paper. It conatins 500 games, each game is a dictionnary with 5 entries:
* **seasons**: a list of the game seasons
* **game**: unique identifier of the game it comes from
* **betrayal**: a boolean indicating if the relationship ended in betrayal or not
* **idx**: unique identifier of the dataset entry
* **people**: the countries played by the players
The season entry is a dictionnary with 3 entries:
* **season**: a year that gives you a notion of order within the seasons
* **iteraction**: a dictionnary that indicates what actions did the betrayer and victim do to each other respectively. Actions available could be either attack, support or None.
* **messages**: contains all the features that the authors of the "Linguistic harbringers of betrayal" rely on to analyze the messages.  
  
The features are the following: 
* **sentiment**: it contains the values for the positive, negative and neutral sentiments
* **lexicon_words**: contains multpiple entries:  
&emsp; *allsubj*: words to compute the subjectivity feature  
&emsp; *premise*: words to compute the premise feature  
&emsp; *claim*: words to compute the claim feature  
&emsp; *disc_expansion*: words to compute the expansion feature  
&emsp; *disc_comparison*: words to compute the comparison feature  
&emsp; *disc_temporal_future*: words to compute the planning feature  
&emsp; *disc_temporal_rest*: word to compute the temporal feature  
&emsp; *disc_contingency*: word to compute the temporal feature  
&emsp; *n_requests*: contains the number of requests  
* **frequent_words**: the frequent words
* **n_word**': contains the number of words
* **politeness**: containss the politeness of the message
* **n_sentences**: contains the number of sentences

According to paper, the talkativeness is quantified with the number of messages sent, the average number of sentences per message, and the average number of words 
per sentence.

### Real and fake news dataset 
The second dataset is the Fake and real news dataset (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).
This dataset contains two sets, a set of real news and another one of fake news. Both sets contain the same features, which are: 
* **text**: The actual text of the news article. 
* **title**: The title of the article.
* **subject**: Every article is classified in a type of subject either 'Government News' or 'Middle-east' or 'News' or 'US_News' or 'left-news' or 'politics' or 'politicsNews' or 'worldnews'.
*  **date**: date of publication of the article.


## Preprocessing 
The  news datasets  requires  some  preprecessing  before  the analysis.  In  fact,  the  news  contain  a  lot  of  links,  tags  ...that are useless for the linguistic cues analysis thus we delete them.  We  also  map  all  the  news  to  lower  case  letters  to avoid miss-leading the models. We also perform some specific modifications  to  remove  empty  strings,  multiple  spaces...  to ensure  that  we  have  proper  entries  both  for  the  analysis  and the models.

The True news have company's name(Reuters) and locaion of news in the beginning, we remove those to avoid havng bias.
 

## Extracting features
### Sentiments
***coreNLP***

The goal was to reproduce the same sentiment analysis as the ones in the paper. The authors relied on the Stanford sentiment analyser for this task. In the first part of this task we implemented methods to compute the sentiments using the Stanford coreNLP, however these computations appeared to be very time consuming ( it will take more than 2 days for the Fake news dataset only), and since we have very limited time and also limited hardware resources, we decided to limit these calculations to a subset of the datasets to see their behaviour on average.  
The coreNLP sentiment analyser computes the sentiment based on how words compose the meaning of longer phrases and not on words independently. We computed the sentiments of each sentence using it and then took the average of the sentences sentiments to get the sentiments of a given news. This was performed on 3000 Real and Fake news respectively. Since the news are independent, (we estimate that 3000 is quite representative of the entire dataset). 
We show here average number of Sentiments with coreNLP for 3000 Fake and Real news resepectively

![](coreNLP.png?style=centerme)

The Fake news have on average more negative sentiments but less positive sentiments than the True news for the samples that we considered. However the number of neutral sentiments seems to be close for both types. We performed a statistical test to compare the mean values for the neutral sentiments and we found that diffrence between the two types for this feature is not significant.

***TextBlob***  

After some researches, we found that other methods exist to perform sentiment analysis, but they are usually considered less efficient than the Stanford methods, which explains the choice of the authors. This alternative method is part of the TextBlob library that allows to compute the polarity of a text. This last, is much less time consuming, thus we were able to compute the polarity of the entire dataset. However note that while the Stanford analyser computes the number of sentiments (very negative, negative, neutral, positive, very positive) on each sentence, the TextBlob method computes the polarity on an entire text and returns a value in the interval [-1, 1] where values under zero represent negative sentiments, values above zero represent positive sentiments and zero is the neutral sentiment.
After computing the polarity of each news, we split the range [-1, 1] into 5 bins to get the sets of sentiments as we had with the Stanford coreNLP. We present here the number of news by sentiment category

![](polarity.png?style=centerme)

There are more positive and negative news (considering overall sentiment of the news, the polarity!) among the Fake news then among the True news. The fourth plot confirms that the Fake news are more sentimental than the True news that tend to be more neutral.

### Politeness  

To compute the politeness of each news, we used the politeness library which is a port of the Stanford Politeness API that was used by the authors. The politeness classifier takes as input a sentence and it's parses and returns the politeness of that sentence. The politeness of a news is computed as the average of it's sentences politenesses. To compute the parses, we first relied on the annotate method of the Stanford coreNLP that is computed while computing the sentiments, but as we were forced to stop this method at a certain point we had to switch to another method to compute the remaining parses. For this we used TextBlob library.  
We show here the average politeness for the Fake vs True news:

![](politeness.png)

The average politeness of the Fake and True news are very close. We performed a statistical test and found that the diffrence between the mean politeness of the Fake and Real news is significant.

### Talkativeness 
We computed the talkativeness of each news, which consists on the number of sentences and the number of words per news. Here also we started with the CoreNLP annotate method then switched to other methods form the NLTK library. 
We show here the average talkativeness for the Fake vs True news:

![](talkativeness.png)

There is a significant diffrence in the average number of words between the two sets, with the Fake news have a higher value on average. However, for the number of sentences, we can see from the plot that the average values are very close. We performed a statistical test on the number of sentences and found that the diffrence in the number of sentences is not significant.





### Discourse connectors
We weren't able to reproduce the same work done by the authors to extract the discourse connectors due to the lack of information. Thus we made our researches on the internet to either find predefined methods that do the task or collect the different markers for the feature and compute the number of their occurrences in the news.
* **Subjectivity**  
for  the  subjectivity  we  used a predefined method from TextBlob library that computes it for a given text. It returns a float in the range [0.0,  1.0]  where  0.0  is  very  objective  and  1.0  is  very subjective.
we show here the results we got for the average subjectivity for the Fake and Real news:

![](subjectivity.png)

The Fake news are on avergae more subejctive than the True news.

* **Expansion, contingency and comparison**  
For these features, no predefined method was found. Thus we collected markers from the internet for each of them and combined them with the features that we extracted from the diplomacy dataset, to get the complete set of markers.
we show here the results we got for the average values of the expansion, contingency and comparison features for the Fake and Real news:

![](discourse_markers.png)

On avergae Fake news contain more expansion, contingency and comparaison discourse connectors than the True news. This shows that True news are less eloquant than Fake news.

## Models 
(nizar)
## Conclusion

   
 
      
        
