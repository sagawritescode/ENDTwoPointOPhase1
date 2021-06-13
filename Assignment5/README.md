# Late Assignment 5 


## Forming the training dataset

There were 3 files
- datasetSentences.txt
- dictionary.txt
- sentiment_labels.txt

Phrase ID for the sentences were supposed to be found in dictionary and then mapped to their respective sentiments. Then sentiments had to be classified in 5 classes.
As Assignment 7 part 1 is specifically dedicated to data preparation with the same dataset, I decided to use readily available PyTreeBank library which simplified data preparation
by avoiding the above steps. Library: [link](https://github.com/JonathanRaiman/pytreebank/tree/master/pytreebank)

The train-test split was 75-25

### Data Augmentation


Used the existing random_delete, random_swap and backtranslate functions. Was iterating over a list and augmenting data with 2 random variables - one for 
sampling the list data and other for deciding how to augment out of 3 methods. But sampling was done properly as the use of np.random for every iteration
does not guarentee absolute probability. Later after some searching, used random.sample which gave the desired sampling


Before moving forward to the strategy lets see a rough distribution of the labels (output for a 75-25 split for a train dataset):

``` frequency distribution:  Counter({1: 2399, 3: 2305, 2: 1684, 4: 1386, 0: 1117}) ```

![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/images/class%20imbalance.png)

Label Percentages

```
1 0.2698
3 0.2592
2 0.1894
4 0.1558
0 0.1256
```


#### Data Augmentation v0

Used a uniform data augmentation strategy. Sample 40% of the dataset without any bias

The accuracy varied from 30 to 33% (Did not save the training logs unfortunately)

#### Data Augmentation v1

There is clearly a class imbalance here. The train data was divided as per the label and the data used for augmentation for lower frequency classes (0,2,4) was more than higher frequency classes (0, 3)

| Label       | Fraction    |
| ----------- | ----------- |
| 0           | 0.6         |
| 1           | 0.1         |
| 2           | 0.3         |
| 3           | 0.1         |
| 4           | 0.5         |

The accuracy touched 37% consistently after this

#### Data Augmentation v2

As no amount of changes in model changed the accuracy significant (not even 2-3%, model changes explained below in detail), we decided to go bonkers and heavily augment the lower frequency classes. Below are the weights

| Label       | Fraction    |
| ----------- | ----------- |
| 0           | 1.0         |
| 1           | 0.2         |
| 2           | 0.5         |
| 3           | 0.2         |
| 4           | 0.8         |


## Model

Decided to use LSTM as the RNN cell for my sentiment analysis. My group members tried with RNN and GRU and the accuracy was nearly the same, so I did not bother trying out the models with them

### Model v1: 

Below is a rough diagram:

![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/images/model%20v2.png)

### Model v2: 

We speculated that final network from the lstm output layer (in 3 digits) to the fully connected output layer was very dense and something might be lost in translation there. So thought of adding a buffer fully connect layer of 25 nodes (25 because the original number of classes were also supposed to be 25)

Below is a rough diagram:

![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/images/model%20v2.png)

___Model v1 over model v2___

The accuracy decreased by 1-2% for the model v2. Hence decided to choose model 1 as the final model

Have attached training logs as files for v1 and v2: link (the train dataset might be different due to shuffling)


### Output/This is what we came for

__Model 1 vs Model 2__:

I decided to choose model v1 over v2 as v2 was not giving any significant improvement over v1. Below is the summary of the comparision between model v1 and v2. Note that these things were constant for all the runs: learning rate = 1e-4, no of lstm cells = 2, epochs = 200 and the data augmentation used was v1 (saner weights for the imbalanced class)


![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/images/summary%20of%20v1%20vs%20v2.png)

v1 training file log - [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/training%20logs/one_linear_layer)
v2 training file log - [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/training%20logs/2_linear_layer)

#### Final Model

I tried different embeddings (100,300, 400) and hidden layer nodes (50, 100, 150) respectively but the output did not change significantly (less than 1%). No training logs were saved for the final model for each combination, but the same numbers were run for the above (the file links can be referred) and this model and I decided to go with the hyperparameters: embedding = 300 and hidden = 100

__A 5% jump with class balancing via data augmentation__:
On model diagnosis (explained in later section), I observed that the model was not predicting positive labels (3 and 4). 3 was getting predicted at some instances but 4 was absent totally from the predictions (again no logs as this analysis was carried way earlier even before v1 and v2). Initially, I thought data augmentation could not be a problem as the label 0 has the least frequency in dataset yet the model predicted 0 heavily. But on doing data augmentation, first by v1 and v2 a 3% and 5% jump in validation accuracy was observed. Though the 100% or 80% data augmentation for less frequency classes felt highly illegal, I too have to get maximum marks so decided to go with data augmentation v2 (insane weights for the less frequency class)

Training logs for final model : [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/training%20logs/training%20logs%20final%20model)

At epoch 200, we touched the best accuracy 39.3% but dil mange more. Epochs were increased to 300 and 40% was touched at some points. Below is the graph for the same. But note that after repeating 1-2 time and even increasing epochs to 400 (cause why not) the model did not touch 40, so the models finals accuracy can be said to __39%__: 



Model diagnosis (To be filled):











## Long road ahead/Things to learn:

- plotting tools for making awesome assignment submissions. Also better diagrams for the neural net architecture 
- learning to debug the models and not just blame it on the data


## Questions to the instructor:

- Data Augmentation
  - How much precent of data augmentation is legal? Can we even make the class of one label thrice its size?
  - Does traditional pre-processing like removing stop words etc should be considered for deep learning models also? Read online that is not the best practice as in deep learning the model should understand

- 








