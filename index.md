
## Introduction

Under the scope of EPFL's ADA class project, we will perform a creative extension on the paper Linguistic  Harbingers  of  Betrayal. In this project we will work on the subject of Natural Language Processing (NLP) and more specially on Linguistic cues. As we will present to you in the next section, the authors of the paper worked on the messages of the diplomacy game and extracted some features to do some analysis and then train a simple model to perform a complex task, even for humans, which is detecting betrayal. To extend this work we will first try to train more complex and adapted models for this kind of tasks, like RNNs (see section 5) to see how they perform and draw some conclusions from the results. Then we will extend this work to another and completely different dataset, a dataset of news. The goal is to try and see if the same features can be used for various NLP examples.

## Related Work: The Linguistic Harbingers of Betrayal paper 

Since our work consists on a creative extension for the Linguistic Harbingers of Betrayal paper, we will describe briefly the work done by the authors to put you in the contest.   
The authors worked on the diplomacy game, which is a war-themed strategy game where friendships and betrayals are orchestrated primarily through language. They collected a dataset, that contains 500 games half of them for games that end with an attack and the other half for games without attacks. Each game contains multiple seasons, and in each season the players communicate via messages then perform and action simultaneously. The two sets are matched to get the most accurate results.   They consider an attacker the first player that breaks the friendship. For games without an attack, the attacker is chosen randomly from the two players.   They performed some preprocessing to extract the following features from the messages: sentiments (negative, positive, neutral), politeness, talkativeness, discourse markers (planning, comparison, expansion, contingency, subjectivity, premises, claims). After extracting these features, they generated some plots to see and compare the behaviours of those between the two types of players. After some analysis, they fed these features to a Logistic regression model that achieved an accuracy of 57% at detecting betrayals.   They concluded that the classifier is able
to exploit subtle linguistic signals that surface in the conversation. They  also analysed how these features evolve as we get closer to betrayal to detect imbalances and check how effective are the features at detecting long-term betrayals.

## Data collection 
(nizar do this!)
As we described in the Introduction, our project consists on two part each of them working on a diffrent dataset.  
The first dataset is the Dilomacy game dataset that was provided with the paper. It conatins 500 games, each game contains multiple seasons


The second dataset is the Fake and real news dataset (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset). This dataset contains two sets, a set of Real new

## Preprocessing 
The  news datasets  requires  some  preprecessing  before  the analysis.  In  fact,  the  news  contain  a  lot  of  links,  tags  ...that are useless for the linguistic cues analysis thus we delete them.  We  also  map  all  the  news  to  lower  case  letters  to avoid miss-leading the models. We also perform some specific modifications  to  remove  empty  strings,  multiple  spaces...  to ensure  that  we  have  proper  entries  both  for  the  analysis  and the models.

The True news have company's name(Reuters) and locaion of news in the beginning, we remove those to avoid havng bias.


## Extracting features
**Sentiments**  

The goal was to reproduce the same sentiment analysis as the ones in the paper. The authors relied on the Stanford sentiment analyser for this task. In the first part of this task we implemented methods to compute the sentiments using the Stanford coreNLP, however these computations appeared to be very time consuming ( it will take more than 2 days for the Fake news dataset only), and since we have very limited time and also limited hardware resources, we decided to limit these calculations to a subset of the datasets to see their behaviour on average.  
The coreNLP sentiment analyser computes the sentiment based on how words compose the meaning of longer phrases and not on words independently. We computed the sentiments of each sentence using it and then took the average of the sentences sentiments to get the sentiments of a given news. This was performed on 3000 Real and Fake news respectively. Since the news are independent, (we estimate that 3000 is quite representative of the entire dataset). 

![](Linguistic-cues-for-Textual-classification\plots\avg nb of Sentiments with coreNLP for 3000 news resepectively.png)

After some researches, we found that other methods exist to perform sentiment analysis, but they are usually less efficient than the Stanford methods, which explains the choice of the authors. This alternative method is part of the TextBlob library that allows to compute the polarity of a text. This last, is much less time consuming, thus we were able to compute the polarity of the entire dataset. However note that while the Stanford analyser computes the number of sentiments (very negative, negative, neutral, positive, very positive) on each sentence, the TextBlob method computes the polarity on an entire text and returns a value in the interval [-1, 1] where values under zero represent negative sentiments, values above zero represent positive sentiments and zero is the neutral sentiment.

**Politeness**  

To compute the politeness of each news, we used the politeness library which is a port of the Stanford Politeness API that was used by the authors. The politeness classifier takes as input a sentence and it's parses and returns the politeness of that sentence. We then takes the average to get the politeness of the news. To compute the parses, we first relied on the annotate method of the Stanford coreNLP that is computed while computing the sentiments, but as we were forced to stop this method at a certain point we had to switch to another method compute the remaining parses. For this we used TextBlob library.

**Talkativeness**  
We computed the talkativeness of each news, which consists on the number of sentences and the number of words per news. Here also we started with the CoreNLP annotate method then switched to other methods form the NLTK library. 
**Discourse connectors**  
We weren't able to reproduce the same work done by the authors to extract the discourse connectors due to the lack of information. Thus we made our researches on the internet to either find predefined methods that do the task or collect the different markers for the feature and compute the number of their occurrences in the news
## Models 
(nizar)
## Conclusion

   

  
  
