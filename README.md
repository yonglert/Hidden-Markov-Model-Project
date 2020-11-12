# Team SPGMA Q6-Q7

- Lee Ying Yang A0170208N - Naive 1, Viterbi 1, Forwards Backwards, Cat Predict

- Tang Yong Ler A0199746E - Naive 2, Forwards Backwards, Cat Predict

- Alvin Tan Jia Liang A0203011L - Viterbi 2, Forwards Backwards, Cat Predict

- Lin Da A0201588A - Viterbi 2, Forwards Backwards, Cat Predict

### Q6b)

The accuracy of our predictions is: 0.05951 (4 s.f)

```python
# iter 0 prediction accuracy:    82/1378 = 0.059506531204644414
```

### Q6c)

The accuracy of our predictions is: 0.03338 (4 s.f)

```python
# iter 10 prediction accuracy:   46/1378 = 0.033381712626995644
```

### Q6d)

We found that the log-likelihood decreases at each iteration. The log-likelihood after first 10 iterations:

```python
iteration 1 started
Loglikelihood after iteration: -105241.28368536178
iteration 2 started
Loglikelihood after iteration: -104921.2739348549
iteration 3 started
Loglikelihood after iteration: -104387.7316150435
iteration 4 started
Loglikelihood after iteration: -103537.55277397987
iteration 5 started
Loglikelihood after iteration: -102400.08014266606
iteration 6 started
Loglikelihood after iteration: -101151.87775264782
iteration 7 started
Loglikelihood after iteration: -99953.05166586963
iteration 8 started
Loglikelihood after iteration: -98804.45516270904
iteration 9 started
Loglikelihood after iteration: -97611.07912099807
iteration 10 started
Loglikelihood after iteration: -96311.79174223388
```

### Q7ii)

Examining the output probability, our HMM has learnt that:

1. $State\ s_0$ corresponds to a negative price change from $-6 \le X \lt 0$
2. $State\ s_1$ corresponds to a positive price change from $0\lt X \leq6$
3. $State\ s_2$ corresponds to no price change, $X = 0$

### Q7iii)

Examining the transition probability, out HMM has learnt that:

1. If we observe a negative price change at current timestep, $i.e\ state\ s_0$, the price change at next timestep is likely to be negative $i.e\ state\ s_0$
2. If we observe a positive price change at current timestep,  $i.e\ state\ s_1$, the price change at next timestep is likely to be stationary/zero $i.e\ state\ s_2$
3. If we observe no price change at current timestep, $i.e\ state\ s_2$, the price change at next timestep is likely to be positive $i.e\ state\ s_1$

### Q7iv)

After learning the output probabilities and transition probabilities, given a sequence of $x_1,x_2,\dots,x_n$ first we can use the Alpha value obtained from the Forward Algorithm, where $\alpha \ =\ P(x_1,x_2,\dots,x_{t-1}, x_t, y_t =j)$

$\alpha_t(j) = P(x_1, ... x_t, y_t=j)\\ \ \ \ \ \ \ \ \ \ \ = \sum_i\alpha_{t-1}(i)a_{i,j}b_j(x_t) \ \ \ \ \ \ (x_t\ is\ unknown\ for\ all\ possible\ x_t)$

Let $x_1,x_2,\dots, x_{t-1}$ be the sequence of outputs as provided by the test data, which is the sequence of prices. Thus, $x_t\ and\ y_t=j$ would be the predicted state and output respectively, after the sequence of prices. Since Yt which is the subsequent state after the sequence of prices in the test data and Xt which is the output are unknown, to find the most likely state and output, we can find the set of $x_t\ and\ y_t=j$ that gives maximum probability for $P(x_1, x_2,\dots,\ x_{t-1}, x_t, y_t =j)$, as shown in the argmax equation below:

$~~\argmax_{x_t, j} P(x_1,\dots, x_t, y_t =j) = \argmax_{x_t,j}[\sum_i \alpha_{t-1}(i)a_{i,j}\ b_j(x_t)] \\  (to\ select\ most\ likely\ end\ state\ and\ output\ for\ unseen\ timestep\ t)~~$

### Q7vi)

The average squared error of our prediction is: 0.9572 (4 s.f)

```python
Fractional improvement of loglikelihood 0.000089 <= thresh of 0.0001
Terminating iterations
Final loglikelihood: -76453.8881941226
average squared error for 981 examples: 0.9571865443425076
```
