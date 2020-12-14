
## Introduction

Under the scope of EPFL's ADA class project, we will perform a creative extension on the paper Linguistic  Harbingers  of  Betrayal. In this project we will work on the subject of Natural Language Processing (NLP) and more specially on sentiment analysis. As we will present to you in the next section, the authors of the paper worked on the messages of the diplomacy game and extracted some features to do some analysis and then train a simple model to perform a complex task, even for humans, which is detecting betrayal. To extend this work we will first try to train more complex and adapted models for this kind of tasks, like RNNs (see section 5) to see how they perform and draw some conclusions from the results. Then we will extend this work to another and completely different dataset, a dataset of tweets. The goal is to try and see if the same features can be used for various NLP examples.

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
The  news datasets  requires  some  preprecessing  before  the analysis.  In  fact,  the  news  contain  a  lot  of  links,  tags  ...that are useless for the linguistic cues analysis thus we delete them.  We  also  map  all  the  tweets  to  lower  case  letters  to avoid miss-leading the models. We also perform some specific modifications  to  remove  empty  strings,  multiple  spaces...  to ensure  that  we  have  proper  entries  both  for  the  analysis  and the models.

The True news have company's name(Reuters) and locaion of news in the beginning, we remove those to avoid havng bias.


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
  
