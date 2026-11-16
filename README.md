# RNN sentiment analysis with TensorFlow and Keras

Lab work for CSET312 (Week 11), Bennett University.

The notebook classifies Amazon product reviews as positive or negative. It trains three Keras models and compares them: an ANN baseline, a SimpleRNN with a small hyperparameter sweep, and an LSTM as the bonus part.

## What it does

- Keeps only `reviews.text` and `reviews.rating`. Ratings of 4 and 5 are positive, 1 to 3 are negative (977 positive, 200 negative).
- Cleans the text (lowercase, removes URLs, punctuation and stopwords, keeps "not").
- Turns reviews into integer sequences of length 150 with a 5000-word vocabulary, split 70/15/15.
- Uses class weights because about 83% of the reviews are positive.

## Results saved in the notebook

| Model | Test accuracy | Macro F1 |
|-------|---------------|----------|
| ANN   | 0.870 | 0.718 |
| RNN   | 0.853 | 0.746 |
| LSTM  | 0.864 | 0.729 |

## How to run

1. Install the packages: `pip install -r requirements.txt`
2. Download the Kaggle dataset `yasserh/amazon-product-reviews-dataset` and put `7817_1.csv` next to the notebook. The file is not included here because of its size.
3. Open `CSET312_Lab_Week11_RNN_Sentiment.ipynb` in Jupyter and run all cells.

`Week11_viva_notes.md` has my short notes for the viva.
