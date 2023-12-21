# Notes

## Article details

**Title:** The role of depressive symptoms and symptom dimensions in actigraphy-assessed sleep, circadian rhythm, and physical activity

**Authors:** Sonia Difrancesco , Brenda W. J.H. Penninx, Harriëtte Riese, Erik J. Giltay and Femke Lamers

**Citation:** (Difrancesco et al., 2022)

**Bibliography:** Difrancesco, S., Penninx, B.W.J.H., Riese, H., Giltay, E.J. and Lamers, F. (2022) The role of depressive symptoms and symptom dimensions in actigraphy-assessed sleep, circadian rhythm, and physical activity. Psychological Medicine [online]. 52 (13), pp. 2760–2766.


**Abstract**

**Background**. Considering the heterogeneity of depression, distinct depressive symptom dimensions may be differentially associated with more objective actigraphy-based estimates of physical activity (PA), sleep and circadian rhythm (CR). We examined the association between PA, sleep, and CR assessed with actigraphy and symptom dimensions (i.e. mood/cognition, somatic/vegetative, sleep). 

**Methods**. Fourteen-day actigraphy data of 359 participants were obtained from the Netherlands Study of Depression and Anxiety. PA, sleep, and CR estimates included gross motor activity (GMA), sleep duration (SD), sleep efficiency (SE), relative amplitude between daytime and night-time activity (RA) and sleep midpoint. The 30-item Inventory of Depressive Symptomatology was used to assess depressive symptoms, which were categorised in three depression dimensions: mood/cognition, somatic/vegetative, and sleep. 

**Results**. GMA and RA were negatively associated with higher score on all three symptom dimensions: mood/cognition (GMA: β = −0.155, p < 0.001; RA: β = −0.116, p = 0.002), somatic/vegetative (GMA: β = −0.165, p < 0.001; RA: β = −0.133, p < 0.001), sleep (GMA: β = −0.169, p < 0.001; RA: β = −0.190, p < 0.001). The association with sleep was more pronounced for two depression dimensions: longer SD was linked to somatic/vegetative (β = 0.115, p = 0.015) dimension and lower SE was linked to sleep (β = −0.101, p = 0.011) dimension. 

**Conclusion**. As three symptom dimensions were associated with actigraphy-based low PA and dampened CR, these seem to be general indicators of depression. Sleep disturbances appeared more linked to the somatic/vegetative and sleep dimensions; the effectiveness of sleep interventions in patients reporting somatic/vegetative symptoms may be explored, as well as the potential of actigraphy to monitor treatment response to such interventions.


**introduction:**

"Major depressive disorder (MDD) is a highly prevalent disorder, associated with high disability (Murray et al., 2012). It often has a chronic course (Verduijn et al., 2017) and a third of the patients experience poor treatment outcomes (Gaynes et al., 2009). Despite the challenges in finding consensus regarding classification, diagnosis and treatment, there has been a recent and significant increase in research to identify novel methods to measure, unravel aetiology, and treat depressive disorders. Actigraphy, an ecologically valid method to objectively measure disturbances in sleep, circadian rhythm (CR) and physical activity (PA), has become widely used in depression research."

Studies have shown depressive disorders are associated with lower daily activity and CR amplitude but "association between actigraphy measures of sleep and depression is less clear."

* night-time activity level appears to be higher in patients with dpression but not always shown in actigraphy data
* previous studies show both insomnia and hypersomnia are associated with depression but author's previous results show:
* "...our previous results with actigraphy have found that higher severity of depressive symptom is associated with longer (but not shorter) SD (Difrancesco et al., 2019). Depression is however heterogeneous in its presentation and these results may not be generalizable to all patients."
* Depression is challenging with opposite clinical presentations - some researchers move away from traditional methods of diagnosis (e.g. DSM-5) and favour  'symptom dimensions' using techniques like factor analysis
  * FA (factor analysis) to identify underlying dimensions, factor structures
  * Three major symptom dimensions -> 1. mood/cognition 2. somatic 3. sleep symptom

"Although it is clear that depression is heterogeneous, little research has focused on the role of symptoms dimensions assessed with severity measures in sleep and CR disturbances and physical inactivity as assessed objectively with actigraphy."

**Study's aim - examine association betweeen actigraphy based PA, sleep and CR wiht symptom dimensions**

**Method**
*Sample*: NESDA dataset -> 370 paticipants with available data in the end
*Symptom dimensions* -> self-reported Dutch IDS; total sum score
* Validity and reliability with Cronbach's reliability
* Factor analysis on sum scores to get items in each dimension, e.g. 
  * **mood/cognition**: feeling irritable, interpersonal sensitivity, feeling sad, diminished quality of mood, feeling anxious or tense, diminished capacity of pleasure/enjoyment, diminished reactivity of mood, diminished interest in people/activities, suicidal thoughts, future pessimism, concentration/decision-making problems, selfcriticism and blame, psychomotor retardation, reduced interest in sex, low energy level/fatigability, leaden paralysis
  * **somatic/vegetative**: panic/phobic symptoms, psychomotor agitation, decreased weight, increase in appetite, other bodily symptoms, decrease in appetite, increased weight, constipation/diarrhoea, aches, and pains
  * **sleep**: early morning awakening, problems sleeping during the night, problems falling asleep, sleeping too much

*Actigraphy* -> sample at 30 Hz, raw data with R package (GGIR)
* Sleep assessed as "total SD per night [in hh:mm] and SE per night [%]"
* CR assessed by "relative amplitude (RA) between daytime and night-time activity per day and sleep midpoint [clock time]."
* RA = amplitude between the activity during the day and the night - lower RA means a dampened CR amplitude suggesting lower activity during the day and higher activity during night.
* Sleep midpoint - proxy for preference for morningness/eveningness; later midpoint means a preference for evening chronotype
* EDA showed RA and sleep midpoint have low correlation - discribe different aspects of CR
* PA is expressed as gross motor activity (GMA) per day (milligravity (mg), $1 g = 9.81 m/s^2$)

*Covariates and descriptive variables* -> age, sex, education

*Statistical analyses* -> distributions chacked on normality with QQ plots; non-normal transformed with log-transformation or Box-Cox. Correlation explored with Pearson's cc.

**Results**

* symptom dimensions moderately to strongly correlated 
* "14-day lower GMA and RA between daytime and night-time activity level were significantly associated with all three symptom dimensions"
* "associations with actigraphy-based sleep were more pronounced for two dimensions"
  * "Longer SD was significantly associated with higher somatic/vegetative symptom dimension score (Table 2; β = 0.113, p = 0.021) but not with mood/cognition and sleep symptom dimension scores." 
  * "Lower SE was associated with higher sleep symptom dimension score (Table 2; β = −0.101, p = 0.011) but not with mood/cognition and somatic/vegetative symptom dimension scores."
  * [...]

**physical inactivity is a feature of depression**
"While few studies have suggested that actigraphy may serve as an objective measure of psychomotor retardation in patients with depression (Krane-Gartiser, Henriksen, Vaaler, Fasmer, & Morken, 2015), our results seem to support that physical inactivity is a general feature of depression (Burton et al., 2013) and symptom severity (Minaeva et al., 2020)."

**patients with depression are typically less active**
"Patients with depression are typically less active and they experience a range of barriers to engaging in PA such as depressive symptoms, higher body mass index, physical co-morbidity, and lower self-efficacy (Vancampfort et al., 2015)."

"As persons with depression are less active than the general population, they encounter additional risks in developing cardiovascular and chronic diseases and mortality."

**behavioural activation may be of benefit to depressive patients**
"As we observed lower PA across all symptom dimensions, PA and behavioural activation may be of help to all patients with depression as it may produce antidepressant effects through multiple biological and psychosocial pathways."

**self-reported sleep v objective sleep is problematic**
"When studying sleep, it is important to make a clear distinction between subjective and objective sleep measures. The first provides information about a person’s perception of his/her sleep quantity and quality. The latter refers to the objective measure of SD and SE with actigraphy. When assessing subjective sleep, persons diagnosed with depression often report sleep disturbances such as insomnia and hypersomnia (Nutt et al., 2008; van Mill et al., 2010)."

**sleep disturbance can be a result of being less active in the first place**
"We found a possible explanation for these findings. Persons with more somatic/vegetative symptoms experience more physical complains, such as aches and pains, bodily symptoms, than may directly impact on their ability to be physically active and resulting in more sleep disturbances. The sleep dimension itself may be problematic as it combines opposite clinical presentations (i.e. insomnia and hypersomnia) for which the associations with objective sleep is expected in opposite directions."

**limitations**
* adjusted for multiple testing but looked at several associations - could result in chance findings
* asssessment of depression symptoms and actigraphy not done at the same time (23 day median between)
* sleep and CRs are complicated - actigraphy may only give indirect and rough estimate; polysomnography is gold standard
* sleep midpoint evaluated during the entire period
* early day/night shifts could have occurred and had an impact - unknown
* classification decisions - e.g. low engergy level part of mood/cog but could be somatic/veg dimension
* did not investigate anxiety

"As lower PA level and dampened circadian amplitude were associated with higher scores on all three dimensions, lower PA level and dampened CRs appeared to be general indicators of depression and depression severity."

"Disturbances in objective sleep were more pronounced for somatic/vegetative symptoms, suggesting that longer SD may be more closely linked to this dimension."

"Sleep interventions may focus not only on patients reporting sleep problems but also on those reporting somatic/vegetative symptoms." 

"PA, behavioural activation, and CR interventions may be not limited to groups of patients based on their symptoms but may be promoted in all patients with depression."

## Data

**Data availability statement.** According to European law (GDPR) data containing potentially identifying or sensitive patient information are restricted; our data involving clinical participants are not freely available in a public repository. However, data are available upon request via the NESDA Data Access Committee (nesda@ggzingeest.nl).