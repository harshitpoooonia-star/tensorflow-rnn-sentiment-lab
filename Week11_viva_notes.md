# Week 11 viva notes: RNN sentiment analysis

## What I did, in 30 seconds

Took the Amazon reviews dataset, kept review text and rating, labelled 4-5 stars as positive and 1-3 as negative (977 vs 200). Cleaned the text (lowercase, removed URLs, punctuation and stopwords but kept "not"), turned it into integer sequences of length 150 with a 5000-word vocabulary, and split 70/15/15. Trained an ANN baseline, a SimpleRNN with a hyperparameter sweep, and an LSTM as the bonus. Used class weights because the data is 83% positive.

## Results

- ANN: 0.870 accuracy, 0.718 macro F1, finds 12/30 negative reviews, 6 s
- RNN (emb 32, hidden 64, dropout 0.3, batch 64, 15 epochs, adam): 0.853 accuracy, 0.746 macro F1, finds 18/30 negatives, 18 s
- LSTM: 0.864 accuracy, 0.729 macro F1, best validation accuracy 0.898, 30 s

Main point: the ANN wins on raw accuracy only because it mostly says "positive". The RNN is better at catching negative reviews, which is what macro F1 shows.

## Likely questions

**Why RNN for text?** Text is a sequence. An RNN reads one word at a time and keeps a hidden state, h_t = tanh(W_x x_t + W_h h_(t-1) + b), so earlier words affect how later words are read. "not good" and "good" give different states.

**What does the ANN lose?** It averages all word vectors (GlobalAveragePooling), so word order is gone. "not good, bad" and "bad, not good" look identical.

**What is an embedding layer?** A lookup table that turns each word id into a trainable dense vector (16 or 32 numbers). Similar words end up with similar vectors. Better than one-hot, which is 5000-dimensional and sparse.

**Why padding and mask_zero?** Reviews have different lengths. Short ones are padded with 0 up to 150, long ones are cut. mask_zero=True makes the RNN ignore the padding.

**Why class weights / macro F1?** 83% of reviews are positive. Always predicting positive gives 83% accuracy and is useless. Class weights make a mistake on a negative review cost about 5x more. Macro F1 averages F1 of both classes equally.

**Why did hidden size 256 fail (0.559)?** Too many parameters for 823 training reviews, and a SimpleRNN with 150 steps suffers from vanishing/exploding gradients. Training never settled.

**Vanishing gradient?** Backprop through time multiplies by the recurrent weight matrix once per time step. If those factors are below 1, the gradient shrinks towards zero for early words, so the RNN forgets the start of long reviews.

**How does LSTM fix it?** It adds a cell state c_t and three gates. Forget gate decides what to drop, input gate what to add, output gate what to expose. The cell state is updated by addition (c_t = f*c_(t-1) + i*g), so gradients can pass through many steps without shrinking. Cost: about 4x the parameters of a SimpleRNN and slower training.

**Effect of dropout?** Randomly switches off units during training to reduce overfitting. Here it barely changed results; the model overfit anyway (train acc ~99%).

**Effect of epochs?** Past 15 epochs test loss went up (0.43 to 0.525) while train accuracy stayed ~100%. That is overfitting. Early stopping would fix it.

**Adam vs RMSprop?** Both adapt the learning rate per parameter. Adam also keeps momentum (a running mean of gradients). Here they were within noise of each other.

**Batch size?** Smaller batches give noisier gradient updates. 32 was worst here (0.768); 64-256 were similar.

**Why not try all 576 combinations?** Too slow for a lab. I varied one hyperparameter at a time from a base config, which still covers every value listed in the brief.

**Loss function?** Binary cross-entropy, with a sigmoid output giving P(positive). Threshold 0.5.

**What would you improve?** More data, pretrained GloVe embeddings, a bidirectional LSTM, and early stopping on validation loss.
