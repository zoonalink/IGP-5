# Notes - Depresjon

## Article details

**Title:** Depresjon: A Motor Activity Database of Depression Episodes in Unipolar and Bipolar Patients

**Authors:** Enrique Garcia-Ceja, Michael Riegler, Petter Jakobsen, Jim Tørresen, Tine Nordgreen, Ketil J. Oedegaard, Ole Bernt Fasmer

**Citation:** (Garcia-Ceja et al., 2018)

**Bibliography:** Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T., Oedegaard, K.J. and Fasmer, O.B. (2018) Depresjon: a motor activity database of depression episodes in unipolar and bipolar patients. In: Proceedings of the 9th ACM Multimedia Systems Conference [online]MMSys ’18: 9th ACM Multimedia Systems Conference. Amsterdam Netherlands, ACM, pp. 472–477. Available from: https://dl.acm.org/doi/10.1145/3204949.3208125 [Accessed 2 December 2023].

## Notes

### Summary

* Wearable data is very common but sensor data in medicine field is rare - non-public, unavailable.
* This paper presents a dataset with sensor data from patients with depression.
* 23 unipolar and bipolar depressed patients and 32 healthy controls
* Each patietn has several days of continuous measuring and demographic data
* Depression is label on MADRS 

### Introduction

**Sensor data availability:**

"There is an increasing awareness in the field of psychiatry on how these activity data relates to various mental health related issues such as changes in mood, personality, inability to cope with daily problems or stress and withdrawal from friends and activities."

**Depression frequency and growth:**

"Since 2010 mental health related problems are the main cause for years lived with disability worldwide. Depression is number one of the most frequent disorders and the current trend is indicating that the prevalence will increase even more in the coming years . Dealing with depression can be demanding since it can create physically, economically and emotionally problems often leading to problems with work and sick leaves."

**Relations betwen sensor data and mood are not well understood:**

"These are complex systems and, because relations between the sensor data and the mood are not well understood yet, changes within these systems are difficult to detect."

"Depression and bipolar disorder are episodic mood disorders, where the pathologic state and the healthy state might be understood as representing different stable states separated by sudden changes."

**Activity level differences:**

"Evidence indicates that a depressive state is associated with reduced daytime motor-activity, as well as increased nighttime activity when comparing to healthy controls. Reduced motor-activity is likewise reported in bipolar depressions, besides increased variability in activity levels compared to others."

**Paper contributes:**

1. novel, open dataset with sensor data of patients (control, depression)
2. large amount of data from both depressed and non-depressed
3. baseline evaluation with ML algorithms for classifying depressed v nondepresse days

### Data

#### Dataset details

* Actigraphy - sampling frequency 32Hz, movements over 0.05g.  Stored as activity count; proportional to intensity of the movement; one minute intervals.
* 23 unipolar and bipolar patients: 
  * 5 inpatients (hospitalised); 18 outpatients
  * severity measured at start and finish on MASDR scale
* 32 non-depressed (control) - 23 hosptial employee, 5 studetns, 4 former patients
* csv file of actigraph data over time for each person:
  * timestamp (1 minute intervals)
  * date (date of measurement)
  * activity (activity measurement from the actigraph device)
* scores.csv contains: 
  * number (patient identifier), days (number of days of measurements), gender (1 or 2 for female or male), age (age in age groups), afftype (1: bipolar II, 2: unipolar depressive, 3: bipolar I), melanch (1: melancholia, 2: no melancholia), inpatient (1: inpatient, 2: outpatient), edu (education grouped in years), marriage (1: married or cohabiting, 2: single), work (1: working or studying, 2: unemployed/sick leave/pension), madrs1 (MADRS score when measurement started), madrs2 (MADRS when measurement stopped)

#### Medical

* Depression - emptiness, sadness, anxiety, sleep disturbance, loss of initiative/interest.
* Symptoms - reduced energy, concentration problems, worthlessness/guiltiness (feelings), suicidality
* Associated with disrupted biological rhythms
* Affects physical, social, etc. health.

**bipolar v unipolar**

"The main difference between bipolar disorder and unipolar depression is that mania is not present in the latter. Depression and bipolar disorder are genetic disorders, and can best be understood as an internal vulnerability to external circumstances disturbing the biological state."

**depression prevalence**

"The lifetime prevalence of depressions is about 15%, but the incidences of episodes with a severity level not qualifying for a depressive diagnosis are far more prevalent. It is well established that depression is characterized by altered motor activity, and that actigraph recordings of motor activity is an objective method for observing mood"

**MADRS**: 
* score below 10 = absence of depressive symptoms
* above 30 = severe depressive state

#### Dataset overview

* 12.6 days => 693 days (402 control and 291 condition)

### Applications of dataset

* Develop systems to automatically detect depression based on sensor data:
  * ML for depression state classification
  * MADRS score prediction
  * sleep pattern analysis
* Evaluating ML methods and approaches:
  * cost-sensitive classification
  * oversampling for imbalanced problems
* comparing different ML classification approaches - feature based, deep learning baed (CNN, RNN for time series)

### Suggested metrics

* variety of measures
* apply weighting and report weighted average

### Baseline performance

* kNN, Linear kernel SVM, radial Basis function kernal (RBF) SVM, Gaussian Process, Decision Tree, Random Forest, Neural Network, AdaBoost, Naive Bayes, Quadratic Discriminant Analysis, ZeroR

![](/literature/_images/2023-12-25-00-16-46.png)

* Extracted features: 
  * mean activity level, SD, % events with no activity (activity level = 0)
  * Normalised
* Evaluation: 10 fold cross validation
* Linear SVM -> best overall weighted recall, accuracy, MCC and F1 score

**Improvements**:

* features which capture complexity
* time series approach 