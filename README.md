# H1N1-Flu-Vaccine
This project was completed during a Computational Data Analysis course, while pursuing my Master's of Science in Analytics from Georgia Tech. The project aimes to predict the likelihood of an individual receiving the H1N1 and seasonal flu vaccines based on a data challenge on DataDriven.org.

## Problem Statement
Since the rise of the COVID-19, public health experts across the world have pointed to the development of 
a vaccine as a key factor in stemming the spread of the disease. Vaccines provide acquired immunity to 
infectious diseases for individuals, and enough immunizations in a community can further reduce the 
spread of diseases through "herd immunity."<sup>1</sup> While the promise of a vaccine provides hope to infectious 
disease experts on defeating the virus, individuals need to be willing to getting the vaccine. An NPR poll 
recently found that only half of Americans say they will get a COVID-19 vaccine once available. Respondents 
cited concerns about the side effects and fears about contracting the virus as reasons for not wanting to 
receive the vaccine.<sup>2</sup>

To explore public health response to a different but recent major respiratory disease pandemic, 
DataDriven.org is hosting a competition around the likelihood of individuals to receive the H1N1 and 
seasonal flu vaccines. H1N1 is the subtype of Influenza A virus and well known outbreaks of H1N1 strains 
occurred during the 2009 swine flue pandemic as well as the 1918 “Spanish” Flu Pandemic. A vaccine for 
the H1N1 flu virus became publicly available in October 2009. The competition aims to explore how we 
can predict if an individual will get a COVID-19 vaccine by looking at data on the H1N1 and seasonal 
vaccines from data from 2009-2010. A better understanding of how these characteristics are associated 
with personal vaccination patterns can provide guidance for future public health efforts. The goal is to use 
these characteristics to predict how likely an individual is to receive their H1N1 and seasonal flu vaccines.

## Data Source
In late 2009 and early 2010, the United States National Center for Health Statistics conducted the National 
2009 H1N1 Flu Survey. This phone survey asked respondents whether they had received the H1N1 and 
seasonal flu vaccines, in addition to questions about the respondents. These additional questions covered 
their social, economic, and demographic background, opinions on risks of illness and vaccine effectiveness,
and behaviors towards mitigating transmission.
The dataset contains the survey responses from 26,706 respondents and includes 35 features. Of the 35 features 13 are binary, 12 are categorical, 8 are ordinal, 
and 2 are numerical. The dataset includes two response variables which are binary and represent whether or not 
respondents reported having received a H1N1 vaccine or a seasonal flu vaccine. 

Figure 1 and Figure 2 display
the proportion of respondents who received the H1N1 and seasonal flu vaccines respectively. Only 21% of respondents 
received the H1N1 vaccine, while 47% received the seasonal flu vaccine.

 <p align="center">
     <b>Figure 1: H1N1 Vaccine Responses</b>
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/H1N1-Flu-Vaccine/blob/main/Visuals/h1n1_waffle2.png?raw=True" />
  </p>
</figure>



<p align="center">
     <b>Figure 2: Seasonal Flu Vaccine Responses</b>
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/H1N1-Flu-Vaccine/blob/main/Visuals/seas_waffle.png?raw=True" />
  </p>
</figure>

Figure 3 shows a more detailed breakdown of each individuals vaccine responses. From the Sunburst Chart we can see that individuals who did not recieve the H1NI vaccine were more likely to not have recieved the Seasonal Flu vaccine and vice versa. The correlation coefficient between the two response variables is 0.38, so there is slight positive correlation between them.

<p align="center">
     <b>Figure 2: Sunburst Chart of Vaccine Responses</b>
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/H1N1-Flu-Vaccine/blob/main/Visuals/sunburst_chart.png?raw=True" />
  </p>
</figure>

## Methodology

### Data Cleansing and Preprocessing
After exploring the dataset two features, Employment Industry and Employment Observation, were identified that had a significant amount of data missing – nearly 50% of the observations. As a result, they were dropped from the dataset. Several of the other features also had missing observations, but none to the magnitude of the two that were dropped. To remedy the missing observations, I chose to impute the data using a K-Nearest Neighbor imputation algorithm, which fills in missing values based on the k-nearest points measured in Euclidean Distance. Prior to performing the imputation, the 10 remaining categorical variables needed to be converted to integer ordinals. Following the imputation, the categorical variables were then encoded and turned into dummy variables. The final result left the data set with 60 features.

### Feature Selection

The cleaned and imputed dataset contained 60 features and while there are 26,706 it is still possible for overfitting to occur if all of the features were used. Dimensionality reduction was explored through the use of Principal Component Analysis, but since the features were primarily categorical it proved to not be effective, with the top principal components accounting for minimal variation in the data set. Filtering was also used to reduce the number of features to those that had an absolute value of correlation greater than 0.05 between the response variable and features. Figure 3 displays the correlation coefficient for each feature and response variable, for the features with the 10 largest correlation coefficients with the response. Recursive feature selection was also explored to identify the features that had the most importance in each classification algorithm. However, when comparing models built with all features and those with features filtered out or recursively selected the models that were trained on the full set of features outperformed those with less features on the testing set. Ultimately it appears that given the number of observations using all 60 features does not appear to cause overfitting in the classification models.

<p align="center">
     <b>Figure 3: Top 10 Features Correlation Coeficients</b>
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/H1N1-Flu-Vaccine/blob/main/Visuals/Response_Corr.png?raw=True" />
  </p>
</figure>

### Model Selection

Given that the problem is a binary classification problem there were numerous Machine Learning algorithms that could reasonably predict the class. Several classification models were explored initially to identify which had the best performance. The K-Nearest Neighbors (KNN) algorithm was screened out due to poor performance. Since the at the core of the KNN algorithm is a distance metric calculation, categorical features make it more challenging for it to classify data points. A Support Vector Machine classification algorithm was also explored but also screened out due to poor initial performance and it being computational-intensive on a large data set. Ultimately Multi-Layer Perceptron (MLP), Boosting, and Random Forrest algorithms were chosen as candidates to validate and test for this classification problem. The following sections describe the methodology and process for each algorithm. For each algorithm used the data was split into training (80%) and testing (20%) for each of the response variables. Prior to training each classification model the data was standardized by removing the mean and scaling to unit variance i.e. normally distributed with mean zero and unit variance. Performance was measured by the Area Under the Receiver Operator Curve.

### Multi-Layer Perceptron Methodology

An MLP is a type of feedforward artificial neural network. The MLP consists of three layers – an input layer, a hidden layer, and an output layer - that attempt to mimic the neurons of the human brain. Given the large size of the dataset an MLP offered to be a promising algorithm that might avoid any issues of overfitting. To construct the architecture of the MLP, I explored using a different number of hidden layers, but after trial and error and conducting some research I determined that a single hidden layer often performed just as good as an MLP with multiple hidden layers. Cross-Validation was performed on the training set to tune the number of neurons in each hidden layer and the regularization parameter. The average Cross-Validation scores implied that a single hidden layer with 5 neurons and a regularization parameter of 0.1 was optimal given the data set.

### Boosting Methodology

Boosting algorithms convert weak classifiers to strong ones by combining them together to form a weighted ensemble. The combination of classifiers allows the boosting algorithm to be robust and produce substantial improvements in performance compared to the use of one single algorithm. I explored an AdaBoost algorithm and a Gradient Boosting algorithm for the classification problem. Both algorithms were tuned with Cross-Validation to identify the optimal number of trees for the underlying Decision Tree Classifier and the learning rate, which is applied to each tree classifier to shrink its weight. 

### Random Forrest Methodology

Random Forests are another ensemble learning method which randomly construct many decision trees and output the class predicted across all trees. Random Forrest method was chosen since it typically outperforms Decision Tree’s and did so in initial evaluations. Cross-Validation was used to tune the number of trees used in the classifier.

## Evaluation and Final Results

After hyperparameter tuning each of the classification algorithms and identifying the optimal parameters each model was evaluated on its accuracy and Area Under the Receiver Operator Curve (AUC) for the test data. While both evaluation criteria are shown the AUC is the preferred metric for measuring classification performance and is what is used in the DataDriven competition to assess submissions. Table 1 displays the results for predicting how likely individual were to receive their H1N1 vaccine and Table 2 displays the likelihood to receive the seasonal flu vaccine.

<p align="center">
     <b>Table 1: H1N1 Vaccine Prediction Results</b>
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/H1N1-Flu-Vaccine/blob/main/Visuals/H1N1%20Prediction%20Results.png?raw=True" />
  </p>
</figure>

<p align="center">
     <b>Table 2: Seasonal Vaccine Prediction Results</b>
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/H1N1-Flu-Vaccine/blob/main/Visuals/Seasonal%20Flu%20Prediction%20Results.png?raw=True" />
  </p>
</figure>

From Figure 1 we know that the responses to seasonal flu vaccine are more balanced than the H1N1 responses and from the correlation plots in Appendix D – Correlation between Features and Response Variables we can see that the features correlation with seasonal flu vaccine responses are of a greater magnitude, so it is not surprising that the algorithms were able to achieve a higher AUC for predicting seasonal flu responses. The Gradient Boosting algorithms performed best for both Reponses, but all other algorithms were within 1% of the AUC measures.

While developing an algorithm that produces the highest AUC is how the competition was scored, understanding what factors and characteristics influence the decision to receive a H1N1 or seasonal flu vaccine are more relevant to helping researchers predict if an individual will receive a COVID-19 vaccine once developed. To better understand the importance of each feature I extracted the importance or weights of the features from each model and have displayed the top 10 in Table 3 for each vaccine.<sup>3</sup> 

<p align="center">
     <b>Table 3: Feature Importance</b>
  </p>
<figure>
  <p align="center">
    <img src="https://github.com/bwalzer4/H1N1-Flu-Vaccine/blob/main/Visuals/seas_waffle.png?raw=True" />
  </p>
</figure>

One of the most important features for both vaccines was whether or not the individuals doctor recommended they receive the vaccine. Two other important behavioral factors were an individuals perception on whether or not the vaccine was effective and their perceived risk around the vaccine. These factors intuitively make sense but emphasize the importance of healthcare providers recommending individuals receive a vaccine and general vaccine education in influencing individuals to receive their vaccines. Since the COVID-19 pandemic is much more similar in nature to the H1N1 swine flu pandemic of 2009 these features should be explored by researchers when trying to determine the likelihood of an individual getting the COVID-19 vaccine. 

## References and Notes
1. DrivenData. (n.d.). Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines. Retrieved July 29, 2020, from https://www.drivendata.org/competitions/66/flu-shot-learning/
2. https://www.npr.org/sections/coronavirus-live-updates/2020/05/27/863401430/poll-shows-only-a-quarter-of-african-americans-plan-to-get-coronavirus-vaccine
3. The lack of interpretability of MLP models is one of its weaknesses and determining feature importance was proved to be too challenging for this project; therefore, the Neural Network model is excluded from this table. The feature importance for the AdaBoost, Gradient Boosting, and Random Forrest algorithms is based on the weighted average of the total decrease in node impurity for each feature based on a Decision Tree Classifier.
