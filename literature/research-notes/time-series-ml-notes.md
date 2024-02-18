# Notes on Machine Learning approaches in Time Series data

## Recurrent Neural Networks (RNNs)

* can model time-series data
* can be stacked, 
* can be uni-directional or bi-directional
* can use sequence to sequence models to forecaset future points from past data
  * 'dense output layers' to predict class or severity

### Long Short-Term Memory (LSTM) 

* suited for learning long-term dependencies and patterns in sequential data like time series
* `long short-term` - can learn both
* can directly model raw sequential data without preprocessing (i.e. into windows or segments)
* gating mechanisms which regulate flow of information into and out of memory cell
  * useful for multi-day actigraphy data (daily patterns)

**citation** (Hochreiter and Schmidhuber, 1997)

### Gated Recurrent Unit (GRU)

* variant of LSTM - simpler and faster, but maybe less good
* merges a reset gate and update gate - controls what information to keep/discard
* 


More reading: 

**Bibliography**

Chung, J., Gulcehre, C., Cho, K. and Bengio, Y. (2014) Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling [online]. Available from: http://arxiv.org/abs/1412.3555 [Accessed 17 February 2024].

Hochreiter, S. and Schmidhuber, J. (1997) Long Short-Term Memory. Neural Computation [online]. 9 (8), pp. 1735–1780. Available from: https://www.bioinf.jku.at/publications/older/2604.pdf [Accessed 17 February 2024].


**recommended** 

**See notes** [/literature/George-ComplexityDetailsDepressed.md](/literature/George-ComplexityDetailsDepressed.md)

George, S.V., Kunkels, Y.K., Booij, S. and Wichers, M. (2021) Uncovering complexity details in actigraphy patterns to differentiate the depressed from the non-depressed. Scientific Reports [online]. 11 (1), p. 13447. Available from: https://www.nature.com/articles/s41598-021-92890-w [Accessed 28 November 2023].

**see notes**: [/literature/Difrancesco-2020-ActigraphySleepDepression.md](/literature/Difrancesco-2020-ActigraphySleepDepression.md)

Difrancesco, S., Penninx, B.W.J.H., Riese, H., Giltay, E.J. and Lamers, F. (2022) The role of depressive symptoms and symptom dimensions in actigraphy-assessed sleep, circadian rhythm, and physical activity. Psychological Medicine [online]. 52 (13), pp. 2760–2766.



### timeseries tutorial

https://www.tensorflow.org/tutorials/structured_data/time_series

* available in colab, github or notebook
* useful to run through...for a sense of the steps in the process, e.g.
  * data, feature engineering, splitting, normalising
  * windowing
  * models
    * single setp (baseline, linear, dense, cnn, rnn, etc.)
    * multi-step (baseline, single-shot, autoregressive)


### directions

* Arora 2023
  * LSTM high level features + statistical features
  * overlapping sliding window
  * SVM
  * 95.57% accuracy
* Jakobsen 2020
  * Random Forest, Deep NN, CNN
  * statistical features = mean, std, prop 0 activity
* Frogner 2019
  * CNN, 1D 
  * F1 - 0.70, MSE - 4.0

**more advanced CNN to improve on Frogner**
**more advanced RNN to improve on Arora**
**predicting severity**
**looking at `segment lengths`** - see Frogner
**apply SMOTE to CNN model** - see Frogner

"When using Synthetic Minority Over-sampling Technique (SMOTE) as a technique for generating more data, they increased their random forest F1-score to 0.73. An extension of this work could be to attempt to use the same sampling strategies as Garcia-Ceja et al. did on the data passed into our CNN."


### features

* mean
* median
* std
* variance
* kurtosis
* coefficient of variance
* interquartile range
* min
* max
* trimmed mean
* spectral density
* entropy
* skewness
* spectral flatness

* 

## TODO

~~add articles to zotero~~
summarise notes for most recent articles from perspective of next steps
read the dissertation
run the lstm model
notes onto teams