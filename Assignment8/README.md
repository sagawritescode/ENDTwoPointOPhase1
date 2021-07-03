# Assignment 8

- Do not use anything from legacy libraries
- Must use Multi30k dataset from torchtext
- use yield_token, and other code written in the class section

We refactored the [2nd](https://github.com/bentrevett/pytorch-seq2seq/blob/master/2%20-%20Learning%20Phrase%20Representations%20using%20RNN%20Encoder-Decoder%20for%20Statistical%20Machine%20Translation.ipynb) and [3rd](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb) file in this [repo](https://github.com/bentrevett/pytorch-seq2seq)

Refactoring of the 2nd file: [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment8/Assignment_8_2_Learning_Phrase_Representations_using_RNN_Encoder_Decoder_for_Statistical_Machine_Translation.ipynb)

Refactoring of the 3rd file: [link](https://github.com/sagawritescode/ENDTwoPointOPhase1/blob/main/Assignment8/Assignment_8_3_Neural_Machine_Translation_by_Jointly_Learning_to_Align_and_Translate.ipynb)

## Refactor legacy libraries: 
Below are the things that were using legacy and were refactored
- Multi30k: Replaced ```from torchtext.legacy.datasets``` with ```from torchtext.datasets```
- Field:  
  - Tokenizer: We use ```from torchtext.data.utils import get_tokenizer``` which returns a tokenizing function which can be later applied, when passed as arguments
  - Building Vocab: Instead of doing ```Field.build_vocab(data)```, we use ```(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)``` for each language. We use the same ```yield_tokens``` function which was written in the class colab file
  - Instead of using init and eos tokens in Field, we added a beginning and end of token via a torch.concat function 
  - Collate Function - We apply these above transformations to the data (in the same order) in a one place called collate function which is passed to the DataLoader
- BucketIterator: Instead of ```from torchtext.legacy.data import BucketIterator``` we use Multi30k's iterator and then use DataLoader ```from torch.utils.data import DataLoader``` where we pass our colate function. We pass DataLoader to the train and evaluate function from which we take the src and trg sentence

## Group members:
- Sagar Shete
- Pushya Mitra
- Kanchana Gore
