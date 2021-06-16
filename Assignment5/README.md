# Late Assignment 5 


## Forming the training dataset

There were 3 files
- datasetSentences.txt
- dictionary.txt
- sentiment_labels.txt

Phrase ID for the sentences were supposed to be found in dictionary and then mapped to their respective sentiments. Then sentiments had to be classified in 5 classes.
As Assignment 7 part 1 is specifically dedicated to data preparation with the same dataset, we decided to use readily available PyTreeBank library which simplified data preparation
by avoiding the above steps. Library: [link](https://github.com/JonathanRaiman/pytreebank/tree/master/pytreebank)

The train-test split was 75-25

### Data Augmentation


Used the existing random_delete, random_swap and backtranslate functions. Was iterating over a list and augmenting data with 2 random variables - one for 
sampling the list data and other for deciding how to augment out of 3 methods. But sampling was not done as desired as the use of np.random for every iteration
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

Decided to use LSTM as the RNN cell for sentiment analysis. We tried with RNN and GRU and the accuracy was nearly the same

### Model v1: 

Below is a rough diagram:

![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/images/model%20v2.png)

### Model v2: 

We speculated that final network from the lstm output layer (in 3 digits) to the fully connected output layer was very dense and something might be lost in translation there. So thought of adding a buffer fully connect layer of 25 nodes (25 because the original number of classes were also supposed to be 25)

Note that in the below models dropout was implemented after every layer.

Below is a rough diagram:

![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/images/model%20v2.png)

___Model v1 over model v2___

There was no significant difference in the accuracy (details mentioned in next section). Hence decided to choose model 1 as the final model




### Output/This is what we came for

__Model 1 vs Model 2__:

I decided to choose model v1 over v2 as v2 was not giving any significant improvement over v1. Below is the summary of the comparision between model v1 and v2. Note that these things were constant for all the runs: learning rate = 1e-4, no of lstm cells = 2, epochs = 200 and the data augmentation used was v1 (saner weights for the imbalanced class)


![Image](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/images/summary%20of%20v1%20vs%20v2.png)

v1 training file log - [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/training%20logs/one_linear_layer)
v2 training file log - [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/training%20logs/2_linear_layer)

#### Final Model

We tried different embeddings (100,300, 400) and hidden layer nodes (50, 100, 150) respectively but the output did not change significantly (less than 1%). No training logs were saved for the final model for each combination, but the same numbers were run for the above (the file links can be referred) and this model and we decided to go with the hyperparameters: embedding = 300 and hidden = 100

__A 5% jump with class balancing via data augmentation__:
On model diagnosis (explained in later section), it was observed that the model was not predicting positive labels (3 and 4). 3 was getting predicted at some instances but 4 was absent totally from the predictions. Initially, I thought data augmentation could not be a problem as the label 0 has the least frequency in dataset yet the model predicted 0 heavily. But on doing data augmentation, first by v1 and v2 a 3% and 5% jump in validation accuracy was observed. Though the 100% or 80% data augmentation for less frequency classes felt highly illegal, I too have to get maximum marks so decided to go with data augmentation v2 (insane weights for the less frequency class)

Training logs for final model : [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment5/training%20logs/training%20logs%20final%20model)

At epoch 200, we touched the best accuracy 39.3% but dil mange more. Epochs were increased to 300 and 40% was touched at some points. Below is the graph for the same. But note that after repeating 1-2 time and even increasing epochs to 400 (cause why not) the model did not touch 40, so the models finals accuracy can be said to __39%__: 



## Model diagnosis:

We wrote a function ourselves to return the predicted values and their count against the actual values. Later, on seeing the best assignments, we realised that this is called the confusion matrix. Below is the extract of the function we wrote

```
def get_pred_vs_act_per_label(pred_act):
  preds, act = list(zip(*pred_act))
  predictions = []
  actuals = []
  for i in preds:
    _, pred = torch.max(i, 1)
    for j in pred:
      predictions.append(j.item())
  for i in act:
    for j in i:
      actuals.append(j.item())

  assert len(predictions) == len(actuals)

  label_dict = {}
  
  for i in range(0, len(actuals)):
    predic = predictions[i]
    actual = actuals[i]
    if label_dict.get(actual, None) != None:
      in_dict = label_dict[actual]
      if in_dict.get(predic, None) != None:
        in_dict[predic] += 1
      else:
        in_dict[predic] = 1
    else:
      in_dict = {predic: 1}
    label_dict[actual] = in_dict
  print("label distribution: actual vs pred:", label_dict)
  return label_dict
```

Sample output (Model v1 with data augmentation v1 at 20th epoch):

```
pred vs actuals: {2: {0: 319, 1: 113, 2: 60, 3: 63, 4: 1}, 0: {2: 45, 4: 1, 0: 519, 1: 151, 3: 60}, 1: {1: 282, 2: 49, 0: 269, 3: 152}, 4: {1: 68, 0: 275, 3: 36, 2: 22}, 3: {0: 104, 1: 193, 3: 157, 2: 25}}
```
The model is hardly predicting the 4th label. At data augmentation v0 (uniform 40% for all), the model was not predicting 3 and 4th label both and predicting 0 heavily. We came over a stack over flow answer which said that the model might be predicting 0 because the model is predicting all labels with equal probability and argmax is returning zero. We diagnosis using the below code if that is the case but the result was zero. The model was heavily predicting 0 with values as 0.9: 

```
preds, act = list(zip(*preds_actual_tup))
zero_preds = 0
same_probability = 0
for i in preds:
  for j in i:
    maxv, pred = torch.max(j, dim = 0)
    minv, pred = torch.min(j, dim = 0)
    if pred == 0:
      zero_preds += 1
      # we used torch.unique also
      if maxv - minv < 0.1:
        same_probability += 1
      
zero_preds, same_probability
```

On encountering the above problem, we decided to go ahead with data augmentation v2 (insane weights for lower frequency classes) and increased the epochs, as we saw that the model required time to learn 3rd and 4th label. 

## Displaying Output

### 25 examples from validation dataset

```
sentence:  It is philosophy , illustrated through everyday events .
actual_label:  Positive predicted_label:  Very Negative

sentence:  Daughter From Danang reveals that efforts toward closure only open new wounds .
actual_label:  Negative predicted_label:  Very Negative

sentence:  A small movie with a big impact .
actual_label:  Very Positive predicted_label:  Negative

sentence:  ( Reaches ) wholly believable and heart-wrenching depths of despair .
actual_label:  Positive predicted_label:  Negative

sentence:  It 's not just the vampires that are damned in Queen of the Damned -- the viewers will feel they suffer the same fate .
actual_label:  Very Negative predicted_label:  Negative

sentence:  A woozy , roisterous , exhausting mess , and the off-beat casting of its two leads turns out to be as ill-starred as you might expect .
actual_label:  Negative predicted_label:  Very Negative

sentence:  It 's difficult to imagine the process that produced such a script , but here 's guessing that spray cheese and underarm noises played a crucial role .
actual_label:  Very Negative predicted_label:  Neutral

sentence:  For most of its footage , the new thriller proves that director M. Night Shyamalan can weave an eerie spell and that Mel Gibson can gasp , shudder and even tremble without losing his machismo .
actual_label:  Very Positive predicted_label:  Positive

sentence:  A model of what films like this should be like .
actual_label:  Very Positive predicted_label:  Very Negative

sentence:  Offers laughs and insight into one of the toughest ages a kid can go through .
actual_label:  Positive predicted_label:  Very Negative

sentence:  Is it something any true film addict will want to check out ?
actual_label:  Neutral predicted_label:  Very Negative

sentence:  What happens when something goes bump in the night and nobody cares ?
actual_label:  Negative predicted_label:  Negative

sentence:  It dares to be a little different , and that shading is what makes it worthwhile .
actual_label:  Positive predicted_label:  Negative

sentence:  If you like quirky , odd movies and/or the ironic , here 's a fun one .
actual_label:  Very Positive predicted_label:  Very Positive

sentence:  It would be interesting to hear from the other side , but in Talk to Her , the women are down for the count .
actual_label:  Neutral predicted_label:  Negative

sentence:  Ah yes , and then there 's the music ...
actual_label:  Neutral predicted_label:  Negative

sentence:  If you 're not the target demographic ... this movie is one long chick-flick slog .
actual_label:  Negative predicted_label:  Very Positive

sentence:  But watching Huppert , a great actress tearing into a landmark role , is riveting .
actual_label:  Very Positive predicted_label:  Neutral

sentence:  We hate ( Madonna ) within the film 's first five minutes , and she lacks the skill or presence to regain any ground .
actual_label:  Negative predicted_label:  Positive

sentence:  Filmmaker Stacy Peralta has a flashy editing style that does n't always jell with Sean Penn 's monotone narration , but he respects the material without sentimentalizing it .
actual_label:  Positive predicted_label:  Positive

```


### 10 false positive examples

```
sentence:  The movie is silly beyond comprehension , and even if it were n't silly , it would still be beyond comprehension .
actual_label:  Very Negative predicted_label:  Positive

sentence:  One of those decades-spanning historical epics that strives to be intimate and socially encompassing but fails to do justice to either effort in three hours of screen time .
actual_label:  Negative predicted_label:  Positive

sentence:  Alas , getting there is not even half the interest .
actual_label:  Very Negative predicted_label:  Positive

sentence:  It 's quite diverting nonsense .
actual_label:  Negative predicted_label:  Positive

sentence:  ... Liotta is put in an impossible spot because his character 's deceptions ultimately undo him and the believability of the entire scenario .
actual_label:  Negative predicted_label:  Positive

sentence:  After the first 10 minutes , which is worth seeing , the movie sinks into an abyss of clichés , depression and bad alternative music .
actual_label:  Negative predicted_label:  Positive

sentence:  Witless and utterly pointless .
actual_label:  Negative predicted_label:  Positive

sentence:  The filmmakers juggle and juxtapose three story lines but fail to come up with one cogent point , unless it 's that life stinks , especially for sensitive married women who really love other women .
actual_label:  Negative predicted_label:  Positive

sentence:  I hated every minute of it .
actual_label:  Very Negative predicted_label:  Very Positive

sentence:  Why he was given free reign over this project -- he wrote , directed , starred and produced -- is beyond me .
actual_label:  Very Negative predicted_label:  Very Positive

```







## Long road ahead/Things to learn:

- plotting tools for making awesome assignment submissions. Also better diagrams for the neural net architecture 
- learning to debug the models and not just blame it on the data


## Questions to the instructor:

- Data Augmentation
  - How much precent of data augmentation is legal? Can we even make the class of one label thrice its size?
  - Should traditional pre-processing like removing stop words etc  be considered for deep learning models also? Read online that is not the best practice as in deep learning the model should understand








