# hidden Markov Model

**Members:**

1. Tang Yong Ler

2. Alvin Tan

4. Lee Ying Yang

3. Lin Da

# POS Tagging of Tweets

By analyzing the sentiment embedded in tweets, we gain precious insights into commercial interests such as the popularity of a brand or the general belief in the viability of a stock[^](https://ieeexplore.ieee.org/document/8455771). In such sentiment analysis, a typical upstream step is the part-of-speech (POS) tagging of tweets, which generates POS tags that are essential for other natural language processing steps downstream.

My team and I have build a POS tagging system using the hidden Markov model (HMM) as a project for our module. The hidden Markov model in our project was used to generate POS tags from a dataset of tweets as well as modified to model intraday trades and predicting stock price movements. 

## Training Data

We were given a training dataset of tweets where each word was tagged to a state as well as the list of possible states. i.e ("John N", "called V", "Mary N") where N = noun and V = verb.

## Naive Approach

We implemented a function to estimate the output probabilities from the training data using *maximum likelihood estimation (MLE)*

<img src="https://render.githubusercontent.com/render/math?math=P(x=w|y=j)=b_j(w)=\frac{count(y\ =\ j\ \to\ x \ =\ w)}{count(y\ =\ j)}">

In this equation, the numerator is the number of times token <img src="https://render.githubusercontent.com/render/math?math=w"> is associated with the tag <img src="https://render.githubusercontent.com/render/math?math=j"> in the training data, and the denominator is the number of times tag <img src="https://render.githubusercontent.com/render/math?math=j"> appears.

### Potential problems

A potential problem that we foresee is that the training data may not be comprehensive enough and does not contain some token (words) that would appear in the test data.

→ This was handled by using *Laplace Smoothing* to the output probability

<img src="https://render.githubusercontent.com/render/math?math=b_j(w) =\frac{count(y\ =\ j\ \to\ x\ =\ w)\ +\  \delta}{count(y\ =\ j\ +\ \delta\ *\ (num\_words+1)}">

### Prediction and results

We naively obtain the best tag <img src="https://render.githubusercontent.com/render/math?math=j^*"> for a given token <img src="https://render.githubusercontent.com/render/math?math=w"> using this equation:

<img src="https://render.githubusercontent.com/render/math?math=j^*=\argmax_jP(x\ =\ w|y\ =\ j)">

**Results:**

**Naive prediction accuracy: 900/1378 (65.3% 3s.f)**

---

## Improved Naive Approach

We improved on our naive approach by estimating <img src="https://render.githubusercontent.com/render/math?math=j^*"> using:

<img src="https://render.githubusercontent.com/render/math?math=j^*=\argmax_jP(y\ =\ j|x\ =\ w)">

This approach finds the most likely tag given the word itself. To do so, we apply Bayes' Rule:

<img src="https://render.githubusercontent.com/render/math?math=P(y=j|x=w)=\frac{P(x\ =\ w|y\ =\ j)P(y\ =\ j)}{P(x\ = \ w)}">

### Prediction and results

**Results:**

**Improved Naive prediction accuracy: 955/1378 (69.3% 3s.f)**

---

## Viterbi Algorithm

In order to improve on our accuracy, we implemented the Viterbi algorithm using the output probabilities calculated using the MLE approach and transition probabilities from the training dataset.

Viterbi Algorithm computes the best tag sequence:

<img src="https://render.githubusercontent.com/render/math?math=y^*_1,y^*_2\dots,y^*_n=\argmax_{y_1,y_2\dots,y_n}P(x_1,x_2\dots,x_n,y_1,y_2\dots,y_n)">

### Prediction and results

**Results:**

**Viterbi Algorithm Accuracy: 1036/1378 (75.2% 3s.f)**

### Potential problems:

1. twitter usernames may be wrongly classified
2. URLs may also be wrongly classified
3. Inconsistencies when dealing with upper and lower case words

---

## Improved Viterbi Algorithm

After improving on our algorithm by dealing with the potential problems, there was a significant improvement in accuracy in our predictions.

### Prediction and results

**Results:**

**Improved Viterbi Algorithm Accuracy: 1116/1378 (81.0% 3s.f)**

---

## Forward-Backward Algorithm

In reality, these tags are not available in tweets. Therefore, we implemented the Forward-Backward Algorithm to learn about the transition and output probabilities when dealing with datasets without these tags made available to us.

---

## Intraday Stock Price Movements

We adopted the hidden Markov model that we have implemented previously to model intraday trades and predicting stock price movements. 

### Output and Transition probabilities

In our prediction, we have utilised the *Forward-Backward Algorithm* that we have implemented previously to obtain reasonable values of output and transition probabilities. After obtaining the output and transition probabilities, the HMM has learnt some semantics on the representation of the states and the behavior of the states. 

### Prediction and results

We ran our adopted HMM on 981 data points.

In this case, the results were judged using averaged squared error 
<img src="https://render.githubusercontent.com/render/math?math=average\ squared\ error= \frac{1}{N}\sum^N_{i=1}(x_i^{predicted}-x_i^{truth})^2">

**average squared error for 981 data points = 0.957 (3s.f)**
