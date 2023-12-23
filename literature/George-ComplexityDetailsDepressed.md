# Notes

## Article details

**Title:** Uncovering complexity details in actigraphy patterns to differentiate the depressed from the non‑depressed

**Authors:** Sandip Varkey George, Yoram K Kunkels, Sanne Booij & Marieke Wichers

**Citation:** (George et al., 2021)

**Bibliography:** George, S.V., Kunkels, Y.K., Booij, S. and Wichers, M. (2021) Uncovering complexity details in actigraphy patterns to differentiate the depressed from the non-depressed. Scientific Reports [online]. 11 (1), p. 13447. Available from: https://www.nature.com/articles/s41598-021-92890-w [Accessed 28 November 2023].

## Notes

### Summary

* Negative association between physical activity and depression well established but not the *precise characteristics* of physical activity patterns
* Complexity measures may identify unexplored aspects of activity patterns - e.g. extent to which there are reptitite periods of physical activity, diversity in durations, etc.
* Actigraphy data: 
  * 4 weeks (~40000 data points each individual)
  * n = 46 (25 non-depressed, 21 depressed)
* Results: 
  * Lower levels of complexity in actigraphy data from depressed group compared to non-depressed controls (in terms of lower mean duration of periods of recurrent physical activity and less diersity in duration of periods)
  * Diagnosis of depression was not significantly associated with mean activity levels or measures of circadian rhythm stability

**Bi-directional association**

"The association between physical activity and depression is well documented. Group-level studies on levels of physical activity have shown an inverse association between physical activity and depressive symptoms. Longitudinal studies, including intervention studies have shown that physical activity and exercise reduces symptoms and improves mood in individuals suffering from depression. Studies have also shown that the association between depression and physical activity, may be bidirectional."

* healthy people are active closer to middle of day and less active in mornings and evenings
* depressed people are active later in the day than healthy people
  * shifted timing affects sleep quality
  * **BUT** studies have been inconsistent and effect sizes small -> suggests that there are other explanations in the patterns
  * current studies have only examined mean activity level or rhythms with constant periodicity
* this study uses tools from complexity science
  * differentiate random activity spikes (noise) from repeated patterns

"Complexity measures would thus provide an objective way to measure to what extent these different types of physical activity (noise versus repeating activity patterns) are present in people with (risk for) depression. If we get a better understanding of what activity patterns differentiate depressed versus healthy people, this may not only provide more insight in how physical activity relates to depression, but, in the case that such patterns are causal to depression, it may also bring new possibilities for diagnostic tools to evaluate whether the patient exhibits healthy physical activity patterns. Moreover, complexity measures quantify an aspect of physical activity that is not captured by existing methods such as the mean activity levels or non-parametric circadian rhythm variables."

### Methods, Data

* **Sample**: 
  * Mood and Movement in Daily Life (MOOVD) study
  * 20-50 years old
  * 54 participants, paired (1:1 depressed to non-depressed) matched on gender, BMI, smoking status and age.
  * Depression score on Beck Depression Inventory (BDI)-II questionnaire

* **Data preprocessing**:
  * reduced oerall size by resampling - averaging 10 minute bines
  * rank transformation to get uniform amplitude distribution
    * ensures activity count time series not affected by extreme events
    * comparable amplitudes

* **Recurrence quantificaion analysis (RQA)**

![](/literature/_images/2023-12-23-11-02-06.png)

Figure 1. Schematic describing the construction of a recurrence plot. The upper panel shows the series of observations vs time. A region centered on the first data point, with width 2ǫ is shaded in blue. The lower panel shows the corresponding recurrence plot. The elements of the recurrence plot corresponding to the first point are shaded in blue, in the lower panel. When a point falls within the blue rectangle in the upper panel, it is shown as a black point in the lower panel. This analysis is repeated for every point in the time series resulting in the complete recurrence plot. The x and y axes of the recurrence plot represent the time of observation (x axis of the upper panel). Hence, when an observation y(t1) at time t1 and y(t2) at time t2 are within ǫ distance of each other, the point (t1, t2) is marked in black in the recurrence plot.

"A recurrence plot reveals the patterns a system makes when it revisits the same neighborhood of space. When the dynamics of a system is purely stochastic, the recurrence plot shows no discernible patterns. On the other hand, when the system shows deterministic behavior the recurrence plot shows distinct patterns in the form of horizontal and diagonal lines. These are quantified using RQA."

* The RQA in this paper is conducted using the free standalone software, TOCSY(Available from tocsy.pik-potsdam.de)

* **Diagonal lines**: "associated with the level of determinism in the time series, since random processes will show these structures very rarely, whereas deterministic processes tend to show these structures more"
  * average of the diagonal line distribution shows the average duration of recurring physical activity patterns in a time series
  * entropy quantifies the diversity associated with the diagonal structures in the recurrence plot
  * (determinism) "DET measure reflects the ratio of points that form diagonal structures to the ratio of all recurring points. Thereby, it provides an estimate of how often different parts of a time series co-evolve as a fraction of the total number of data point pairs in the plot"
* **Vertical lines**: "indicate periods of “stasis” or very slow evolution. In a sense, it shows the length of the activity, with longer vertical lines suggesting an activity that lasts for longer."
  * the mean of the vertical line distribution shows the mean levels of stasis associated with the physical activity patterns (i.e how long an activity persists) 
  * entropy yields the diversity associated with the vertical line distribution
  * (laminarity) "LAM measure reflects the ratio of points that form vertical structures to the ratio of all recurring points. This provides an estimate of how often slowly evolving processes occur, as a fraction of the total number of data point pairs"

![](/literature/_images/2023-12-23-11-21-04.png)

* **Missing data**
  * Removed

* **Traditional actigraphy quantifiers**
  * IS - interdaily stability
  * IV - intradaily variability
  * RA -relative amplitude

Circadian measures were calculated using the ACTman package in R version 3.6.3

* **Statistical analysis**
  * t-test for group differences in complexity
  * t-test of independent samples - identical means
  * Welch t-test (no assumption of equal population variance), generalises to unequal sample sizes
  * Cohens d for effect size
  * Spearman rank correlation coefficient to calculate independence between different measures
  * logistic regression to predict diagnostic status with traditional actigraphy quantifiers and recurrence quantifiers

![](/literature/_images/2023-12-23-11-21-57.png)

### Results

![](/literature/_images/2023-12-23-11-24-01.png)

![](/literature/_images/2023-12-23-11-24-37.png)

![](/literature/_images/2023-12-23-11-25-07.png)

* No differences in demographic and clinical characteristics becuase of initial pairing
* No significant differences betweeen groups when looking at traditional actigraphic quantifiers
* **BUT** mean differences in recurrence plot parameters is significant: 
  * mean and entropy of diagonal line length distribution (depressed group lower mean, entropy)
  * ratio of LAM to DET (depressed group higher ratio)

![](/literature/_images/2023-12-23-11-31-01.png)

* Overall, complexity measures were significant, borderline significant (p<.1) and in expected direction
* all quantifiers (except LAM) showed medium to large effect sizes
* Bonferroni correction in level of alpha results in all significance los
* logistic regression prediction better than traditional actigraphy measures

![](/literature/_images/2023-12-23-11-34-34.png)

"We have some recommendations for future research. First, it is relevant to explore whether changes in the complexity of physical activity patterns actually precede the onset of symptom changes in individuals with depression. RQA has been used to successfully predict sudden transitions in other fields. The Transitions In Depression (TRANS-ID) study has collected unique personalized datasets in which people are followed intensively over the course of symptom transitions, including actigraphy measurements. This is therefore the ideal design to test whether decreases in the complexity of physical activity precedes symptom transitions in depression. Another recommendation for future studies is to examine whether intervention on physical activity patterns in the direction of increased complexity in depressed patients would lead to a reduction in the level of symptoms."

"It is concluded that the diversity and average duration of activities was significantly associated with depression, while mean levels in physical activity and circadian rhythm variables were not. This novel finding has important implications for understanding how physical activity relates to mood disorders like depression. If future studies will replicate this finding and show support that complexity patterns causally relate to development of symptoms, RQA measures may constitute an additional tool for personalized diagnostics and treatment strategies, in depression."
## Data Availability

However, data are available from the corresponding author on reasonable request. The codes used for data analysis in this paper may be found at github.com/sgeorge91/ComplexityInDepression.