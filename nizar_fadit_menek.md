

## Introduction

Under the scope of EPFL's ADA class project, we will perform a creative extension on the paper Linguistic  Harbingers  of  Betrayal. In this project we will work on the subject of Natural Language Processing (NLP) and more specially on sentiment analysis. As we will present to you in the next section, the authors of the paper worked on the messages of the diplomacy game and extracted some features to do some analysis and then train a simple model to perform a complex task, even for humans, which is detecting betrayal. To extend this work we will first try to train more complex and adapted models for this kind of tasks, like RNNs (see section 5) to see how they perform and draw some conclusions from the results. Then we will extend this work to another and completely different dataset, a dataset of tweets. The goal is to try and see if the same features can be used for various NLP examples.

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
&emsp;* *allsubj*: words to compute the subjectivity feature
&emsp;* *premise*: words to compute the premise feature
&emsp;* *claim*: words to compute the claim feature
&emsp;* *disc_expansion*: words to compute the expansion feature
&emsp;* *disc_comparison*: words to compute the comparison feature
&emsp;* *disc_temporal_future*: words to compute the planning feature
&emsp;* *disc_temporal_rest*: word to compute the temporal feature
&emsp;* *disc_contingency*: word to compute the temporal feature
&emsp;* *n_requests*: contains the number of requests
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
* **date**: date of publication of the article.



## Preprocessing 
The news that we got contain tags and links... that are irrelevant 

## Extracting features
**Sentiments**

**Politeness**

**Talkativeness**

**Discourse connectors**

## Models 

## Conclusion



```markdown
Syntax highlighted code block
   
# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/zeineb111/data_story_WAP/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
