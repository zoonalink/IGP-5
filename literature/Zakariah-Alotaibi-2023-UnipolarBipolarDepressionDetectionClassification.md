# Notes

## Article details:
**Title:** Unipolar and Bipolar Depression Detection and Classification Based on Actigraphic Registration of Motor Activity Using Machine Learning and Uniform Manifold Approximation and Projection Methods

**Authors:** Mohammed Zakariah and Yousef Ajami Alotaibi

**Citation:** (Zakariah and Alotaibi, 2023)

**Bibliography:** Zakariah, M. and Alotaibi, Y.A. (2023) Unipolar and Bipolar Depression Detection and Classification Based on Actigraphic Registration of Motor Activity Using Machine Learning and Uniform Manifold Approximation and Projection Methods. Diagnostics [online]. 13 (14), p. 2323.

# Summary

## Dataset
The dataset utilised in the study comprises actigraphic recordings of motor activity from individuals with schizophrenia (both unipolar and bipolar) and healthy controls. Here's a breakdown of the dataset attributes and data visualisation provided:

**Dataset Attribute**

- **Actigraph Recordings:** Actiwatches worn on the participants' right wrists were used to record motor activity, providing information on sleep habits, rest states, and overall activity levels.
- **Personal Information:** Includes participant demographics such as age, gender, education, employment, marital status, type of affiliation, and presence of melancholy.
- **MADRS (Montgomery–Asberg Depression Rating Scale):** Provides standardised measures of depressive symptoms, allowing for the investigation of correlations between depressive symptoms and motor activity patterns.

**Data Visualisation**

- **Age Distribution:** Figure 3 illustrates the age distribution of participants, indicating a range from 20 to 69 years old, with a higher number of participants in the age groups of 45 to 54.
- **Activity Sample:** Figure 4 displays a data sample from actigraphy recordings, showing the activities of both healthy participants and those exhibiting various forms of depression.
Recording Duration: Figure 5 shows the activity logged over 24 hours by participants in the control and condition groups, with data compiled for bipolar I, bipolar II, and unipolar depression.
- **Dataset Structure:** The dataset is split into directories containing recordings from the control and condition groups, with CSV files containing the date, timestamps, activity measures, and scores for each participant.

**Additional Information**

- **Data Collection Frequency:** Data were collected every minute at a frequency of 32 Hz for movements greater than 0.05 g.
- **Recording Duration:** Most recordings spanned over 13 days, with variations in recording durations for individual subjects.

## Methodology
**Leave-One-Out Validation Technique:**
To prevent overfitting, a Leave-One-Out validation technique is employed. This involves training the classifier using data from all participants except the one being tested, ensuring unbiased evaluation of the classifier's performance.

**Class Imbalance Handling:**
Due to class imbalance in the dataset, two oversampling methods are utilised: random oversampling and SMOTE (Synthetic Minority Over-Sampling Technique). These techniques balance the class distribution, providing the classifier with a more representative training dataset.

**Data Preprocessing:**
The data undergo various preprocessing steps to ensure quality and compatibility with the system. This includes data cleaning to handle missing or erroneous values, imputation techniques to fill missing values, outlier detection, and numerical transformation of variables.

**Feature Extraction:**
It involves creating feature vectors from statistical data extracted from each participant's daily activity records. The dataset is divided into four arrays representing different classes: healthy, bipolar I, bipolar II, and unipolar depression.

**Dimensionality Reduction with UMAP:**
Unsupervised machine learning dimensionality reduction using UMAP (Uniform Manifold Approximation and Projection) is applied to improve performance and visualize high-dimensional data in a lower-dimensional space.

**Model Training:**
The data, after dimensionality reduction, are split into training, testing, and validation sets. The neural network model is trained using the Adam optimiser and cross-entropy loss function. Performance metrics such as accuracy score, F1-score, and Cohen Kappa are used to evaluate the model.

**Hyperparameter Tuning:**
Hyperparameters of the model are optimised using grid search, random search, and Bayesian optimisation techniques. This iterative process aims to find the best combination of hyperparameters that maximise the model's performance and generalizability.

**Workflow Diagram and Visualisation:**
The workflow diagram outlines the implementation process, including data processing, feature extraction, and model training. Visualisation techniques such as plotting the age distribution of participants and activity samples aid in understanding the dataset characteristics.

## Results
**Experiment 1:**

Using unprocessed data, the model achieved moderate performance metrics due to class imbalance issues.
The training score surpassed 0.8 in accuracy and F1-score, but the validation score remained around 0.6, indicating poor generalization.
The Cohen Kappa validation score was less than 0.1 due to class imbalance, indicating low agreement beyond what would be expected by chance.

**Experiment 2:**

The model reached high accuracy (around 0.991) and F1-score (0.99) with the help of UMAP dimensionality reduction.
Most scores peaked around 20 epochs, indicating rapid convergence and high efficiency of the model.
The Cohen Kappa validation score significantly improved to 0.977, demonstrating strong agreement beyond chance.

**Comparison with Other Studies:**

Comparison with other studies revealed that the combination of UMAP and neural networks produced superior outcomes compared to other machine learning algorithms.
Previous studies achieved lower scores, indicating the effectiveness of the proposed approach.

**Confusion Matrix Analysis:**

The confusion matrix highlighted class imbalances and misclassifications, particularly in the Bipolar II class due to limited sample size.
Samples labeled as healthy were most prevalent, while Bipolar II samples were scarce, impacting classification accuracy.

**Additional Statistical Analysis:**

Various statistical measures, including mean, number of zeros, skewness, and standard deviation, provided insights into patient characteristics and behaviors.
Patients with depression appeared to be more active than average, and MADRS scores generally decreased significantly over time, possibly indicating improved treatments.

**Future Directions:**

Future studies could explore classification beyond depressed and non-depressed classes, considering traits shared by misclassified patients.
Gathering MADRS scores for control groups in future datasets would enhance classification performance and provide a better understanding of depression.

## Discussion
**Methodology Overview:**

The study aimed to classify multiple types of mental illness using ML, focusing on motor activity data collected through actigraphy devices.
Personal details and depressive symptoms assessments were included in the dataset, providing rich information for classification.
Two experiments were conducted, one with unprocessed data and another with UMAP dimensionality reduction, in combination with neural networks.

**Results Summary:**

Experiment 1 showed moderate performance due to class imbalance issues and incomplete model training.
Experiment 2, utilizing UMAP and neural networks, achieved high accuracy and F1-score, demonstrating the effectiveness of the approach.
Comparison with other studies highlighted the superiority of the proposed method in depression classification.

**Limitations:**

The study's sample size was relatively small, potentially limiting the generalizability of the findings.
Class distribution imbalance, particularly in the Bipolar II class, could bias the results and affect the model's performance.
Incomplete model training in Experiment 1 may have impacted performance evaluation.
Lack of comparison with gold-standard diagnostic methods for depression limits the validation of ML and UMAP methods.
The focus on motor activity neglects other potentially important features for depression detection.
Lack of external validation and long-term monitoring limits the comprehensive understanding of the relationship between motor activity and depressive symptoms.

**Future Directions:**

Future research should address sample size limitations and class distribution imbalance for more robust findings.
Completing model training and ensuring comprehensive evaluation methods are essential for accurate performance assessment.
Comparison with established diagnostic methods and inclusion of additional features could enhance the validity of ML and UMAP methods.
External validation and long-term monitoring could provide a more nuanced understanding of the relationship between motor activity and depression.


## Conclusion
The paper presents a novel approach to depression detection and classification using a combination of unsupervised machine learning dimensionality reduction, neural networks, and uniform manifold approximation and projection (UMAP). Two experiments were conducted to evaluate the proposed method, one with dimensional reduction and one without, demonstrating its effectiveness in distinguishing between healthy and ill instances as well as different stages of depression.
The results of the experiments showed remarkable performance, with the model achieving high accuracy, F1-score, and Cohen Kappa score. Despite completing the model training process in fewer epochs than anticipated, thanks to the Adam optimizer, there were challenges with the validation score not performing as well as the training score, particularly in the first experiment.
In the second experiment, rapid convergence to a high accuracy of 0.991 was observed, which aligned with the outcomes of UMAP dimensionality reduction. Additionally, various machine learning classification techniques were employed, with the highest score achieved among these algorithms being 0.727. However, the limited number of samples in the Bipolar II class, especially in the testing set, adversely affected the classification performance.
The study suggests several future research directions, including exploring additional dimensionality reduction techniques, integrating multimodal data sources, conducting longitudinal studies, validating the proposed method in clinical settings, and focusing on the interpretability and explainability of depression classification models. These avenues of research have the potential to advance the diagnosis, treatment, and support for individuals with depression.


