# Notes
## Article details
**Title:** An Ensemble Classification Model for Depression Based on Wearable Device Sleep Data

**Authors:** Yuzhu Hu, Jian Chen, Junxin Chen, Wei Wang, Shen Zhao, and Xiping Hu

**Citation:** (Hu et al., 2017)

**Bibliography:** Yuzhu Hu, Jian Chen, Junxin Chen, Wei Wang, Shen Zhao, and Xiping Hu (2017) An Ensemble Classification Model for Depression Based on Wearable Device Sleep Data - PubMed [online]. Available from: https://pubmed.ncbi.nlm.nih.gov/37030745/ [Accessed 8 December 2023].

## Notes
### Summary

**Background and Problem Statement**
- Depression is a prevalent mental disorder, exacerbated by sleep disturbances.
- Wearable devices have gained popularity for tracking sleep quality, offering an opportunity for intelligent and cost-effective depression detection.
- Missing data is a common problem with wearable devices, and existing depression identification studies often rely on complex data, hindering generalization and susceptibility to noise interference.

**Proposed Solution**
- A systematic ensemble classification model for depression (ECD) to address the issues mentioned.
- An improved Generative Adversarial Interpolation Network (GAIN) is designed to handle missing data more effectively, achieving a more reasonable treatment of missing values compared to previous methods.
- Ensemble learning is utilized for depression recognition, combining five classification models (SVM, KNN, LR, CBR, and DT) to enhance robustness and generalization, with a voting mechanism to improve noise immunity.#

**Research Contributions**
- Analysis of depression based on sleep data collected from wearable devices, offering a simple and easily generalizable approach.
- Proposal of an improved data interpolation method based on GAN to address missing data challenges.
- Design of an ensemble learning-based depression classification model for better classification results.
- Evaluation of the ECD model through experiments on real-world datasets, demonstrating its effectiveness in depression detection.

**Related Works**

- Traditional methods for depression identification rely on self-assessment scales and hospital visits, while computer-assisted methods often use EEG data.
- Wearable device research in depression analysis has focused on multi-modal datasets, with few studies specifically targeting sleep data.


### Method
**Sleep Feature Extraction**

- Identification of sleep periods based on activity logs and comparison with healthy controls.
- Selection of sleep-related features, including sleep start time, sleep end time, total duration of sleep, sleep efficiency, length of waking during sleep, number of awakenings during sleep, frequency of waking during sleep, maximum activity during sleep, and minimum activity during sleep.
- Extraction of two weeks of sleep data for each subject, incorporating gender and age as label values.

**Improved GAIN for Missing Data**

- Data preprocessing steps, including missing data processing and standardization.
- Introduction of the GAIN method for missing data imputation, utilizing a generator and discriminator to generate interpolated values.
- Design of a transformation function to re-scale input values and ensure reasonable interpolated values.
- Evaluation of imputed values using Mean Absolute Error (MAE) criterion.
- Utilization of different methods for standardization such as min-max method, Z-scores method, and self-fitting transformation function.

**Classification**

- Building a classification model using ensemble learning with various machine learning classification algorithms: Support Vector Machine (SVM), K Nearest Neighbors (KNN), Logistic Regression (LR), Case-based Reasoning (CBR), and Decision Tree (DT).
- Description of the classification functions for each weak classification model.
- Introduction of Bagging ensemble learning method to combine weak classifiers for final classification results.
- Adjustments made to address imbalances in depressed and non-depressed samples, including adjusting loss function weights and confidence levels in voting results.

### Dataset 
- The dataset used in the study comprises daily activity data collected from individuals with schizophrenia and major depressive disorder.
- Data was collected using wristwatches from Cambridge Neurotechnology Ltd, UK, measuring activity information at a sampling frequency of 32 Hz.
- It includes activity-recording data from both depressed patients and healthy controls, along with additional personal information about the participants.

### Implementation Details
- The effectiveness of the improved GAIN model is evaluated by comparing it with other interpolation methods, including the original GAIN, Lagrange interpolation, and average interpolation.
- An ensemble learning-based depression classification model is constructed using five weak classifiers: SVM, KNN, LR, CBR, and DT. Feature selection methods are employed to simplify features, and parameters for each classifier are optimized.

### Results and Analysis
- The improved GAIN model achieves the best interpolation results for several features, outperforming other interpolation methods.
- The ensemble learning-based classification model achieves high accuracy, with the ensemble classifier performing better than individual classifiers. Results are compared with other methods, demonstrating the superiority of the proposed model.

### System
- A depressive disposition discrimination system is designed based on the proposed ensemble classification model. It collects user sleep data from wearable devices and provides mental health assessments.
- Users can upload their data for analysis through a website interface, and the system delivers prompt messages for depressive tendencies.

### Discussions and Conclusion
- The study highlights the significance of using wearable device data for depression analysis, particularly focusing on sleep disturbances as an indicator.
- The ensemble learning-based classification model demonstrates high accuracy in identifying depression, offering practical implications for mental health assessment in daily life.
- Future directions include collecting more relevant data and enhancing the ensemble system for better mental health advisories.