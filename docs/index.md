# Mean Reversion Trader
Devon O'Quinn, Shayali Patel, Nicholas Nitsche, Julien Perez, Mutimu Njenga

## Introduction
Mean reversion trading is based on the principle that prices tend to revert toward their average after periods of overextension. To develop a consistent strategy, we leverage machine learning to extract predictive signals from market data and optimize performance for low-capital intraday trading.

### Literature Review
There are two notable features to consider for mean reversion trading. VWAP is “the price a ‘naive’ trader can expect to obtain”; buyers should fill below it and sellers above it [1]. The anchored VWAP (AVWAP) is essentially the same, but starts computing VWAP at a user‑set time. Because AVWAP starts computing from a chosen anchor and can measure over any time interval, if the price stays above a rising AVWAP, there is a bullish trend; if the price falls below a declining AVWAP, there is a bearish trend [1]. Bollinger Bands summarize trend and volatility; prices near the upper/lower band indicate overbought/oversold conditions. They serve as a useful secondary indicator since they are not predictive [2]. Using features such as these, ML can be used to implement a statistically sound and consistent trading strategy.

Labels are essential. The triple barrier method gives realistic labels: each event has a profit‑taking level, a stop‑loss level, and a vertical barrier representing a maximum holding period; the event’s label depends on which barrier is hit first [3]. Event‑based sampling (CUSUM) and activity‑based bars regularize information flow and reduce time‑bar heteroskedasticity [3]. Gaussian mixture models assign probabilities under a mixture of Gaussians, and can yield a soft “overextension score” from joint features (VWAP distance, Bollinger position, short‑term momentum, relative volume) [4]. GMMs adapt to regime shifts and heavy tails better than simple z‑scores.

Random Forests average many decision tree classifiers, lowering variance, and are well-suited to nonlinear, correlated data [5]. However, naive bootstrap aggregating can overstate performance when labels overlap; using sequential bootstrap and limiting the number of samples to the average uniqueness of labels is a better method [3]. Probabilities map to position sizes and whether to trade or not to trade.

Gradient‑boosted trees build predictors by correcting residuals and can be adapted to quantile regression, a useful tool for exit sizing [6]. Estimating conditional quantiles (e.g., 10th/50th/90th) allows for dynamic stops and targets; additionally, tree-based quantile methods outperform classical quantile methods in high-dimensional, nonlinear settings. [7].
Lastly, model evaluation requires careful consideration. Traditional random test/train splits fail for financial data because features and labels are serially correlated [3]. Purged k‑fold with embargo removes overlap between samples, and CPCV returns a distribution of out‑of‑sample metrics from purged k-fold, providing a more reliable measure of model performance [3].

### Dataset Description
Intraday U.S. equities from Georgia Tech’s Bloomberg Terminal.

## Problem Definition

### Problem & Motivation
Low capital trading favors low‑turnover, high‑conviction trades. Traditional retail day trading strategies utilize naive heuristic rules that result in bad and inconsistent profit margins; additionally, they are riskier and less flexible. We aim to implement a successful and consistent mean‑reversion strategy through an ML Pipeline.

### Objective
Given an overextension event where price is far from an AVWAP, determine whether the price will revert sufficiently within the next 15–30 minutes to yield a profitable mean‑reversion trade. If a profitable reversion is likely, determine where to set dynamic take‑profit and stop‑loss levels to maximize returns by predicting the distribution of return magnitudes and executing the trade accordingly.

## Methods

### Data Preprocessing
1. **Stock selection.** We will assume a trading capital of $2,500. Stocks will be selected based on the following criteria:
    - Price band: $10–$100, so we can buy multiple lots without over-allocating capital.
    - Liquidity: high average volume and tight bid-ask spreads to keep costs low.
    - Volatility: moderate intraday range.
2. **Data window.** Regular hours, excluding the first and last five minutes. We remove duplicates, halts, and out‑of‑order samples. We will collect data over 5 days.
3. **Bar construction.** Aggregate ticks into volume or dollar bars to normalize information content.
4. **Feature engineering.** VWAP distance (expressed as a z-score), Bollinger position, short‑term momentum, relative volume, time‑of‑day, and a five‑minute recent trading context feature.
5. **Normalization.** Standardize per asset and per session for comparability.
6. **Labeling.** Triple‑barrier labels are used. A trade is successful if the price reverts before the stop or time limit (15–30 minutes).
7. **Splitting.** Purged k‑fold with embargo to prevent information leakage.

### Models
1. **Unsupervised overextension detector.** GMM on event features. Events whose log‑likelihood falls below a threshold are flagged as overextended candidates.
2. **Supervised opportunity classifier.** Random Forest traind on the overextension candidates; it outputs the probability of reversion.
3. **Supervised exit model.** Gradient‑boosted trees (trained on the events predicted to revert with high probability) to predict τ‑quantiles; we will use the 0.1 quantile as the stop loss condition and the 0.5 quantile as the target condition.

## Results & Discussion

### Metrics
For the GMM, we will look at the log-likelihood of events and stability of flagged overextensions. We will report the mean and standard deviation of F1 scores, negative log‑loss from purged k‑fold cross‑validation, and the proportion of events the model chooses to trade for the classifier. For the exit model, we will evaluate the calibration of predicted quantiles and use out‑of‑sample profit‑and‑loss per trade and Sharpe ratio as final metrics. The CPCV procedure produces a distribution of Sharpe ratios, allowing us to compute confidence intervals and assess the stability of the strategy.

### Project Goals
- Build an intraday ML pipeline across multiple assets.
- Outperform naive heuristic trading.

### Expected Outcomes
We expect the strategy to trade only a small number of overextensions, with a success rate significantly above random. We also expect it to perform better than standard heuristic trading.

## Team Logistics

### Gantt Chart
...

### Contribution Table
| Name            | Proposal Contributions        |
|-----------------|-------------------------------|
| Devon O'Quinn   | Gantt Chart, Literature Review |
| Shayali Patel   | Gantt Chart, Objective        |
| Nicholas Nitsche| Gantt Chart, Methods          |
| Julien Perez    | Gantt Chart, Results          |
| Mutimu Njenga   | Formatting                    |

## References
[1] B. Shannon, “Anchored VWAP,” Alphatrends. [Online]. Available: https://alphatrends.net/anchored-vwap/.

[2] C. Thompson, “Understanding Bollinger Bands: A Key Technical Analysis Tool for Investors,” Investopedia, Sep. 3, 2025. [Online]. Available: https://www.investopedia.com/terms/b/bollingerbands.asp.

[3] M. López de Prado, Advances in Financial Machine Learning. Wiley, 2018.

[4] C. M. Bishop, Pattern Recognition and Machine Learning. New York, NY, USA: Springer, 2006, ch. 9 (Mixture Models & EM).

[5] L. Breiman, “Random Forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001. doi:10.1023/A:1010933404324.

[6] J. H. Friedman, “Greedy Function Approximation: A Gradient Boosting Machine,” Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001. doi:10.1214/aos/1013203451.

[7] N. Meinshausen, “Quantile Regression Forests,” Journal of Machine Learning Research, vol. 7, pp. 983–999, 2006.
