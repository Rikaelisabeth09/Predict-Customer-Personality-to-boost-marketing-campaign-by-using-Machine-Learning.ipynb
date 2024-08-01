# Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb
This repository contains the code and analysis for predicting customer personality traits to enhance marketing campaigns using machine learning techniques. The project aims to segment customers based on their behavior and characteristics to optimize marketing strategies and improve campaign performance.

**Table of Contents**
- [Problem Statement](#problem-statement)
    - [Project Overview](#project-overview)
    - [Goal](#goal)
    - [Objective](#objective)
- [Data Preparation](#data-preparation)
    - [Handling Data](#handling-data)
    - [Feature Engineering](#feature-engineering)
- [Data Exploration](#data-exploration)
- [Data Modeling K means](#data-modeling-with-k-means)
    - [Data Preprocessing ](#data-preprocessing)
    - [Modeling](#modeling)
    - [Cluster Segmentation](#cluster-segmentation)
- [Customer Personality Analysis](#customer-personality-analysis)
- [Business Recomendation](#business-recommendation)


# Problem Statement

## Project Overview
"A company can grow rapidly by understanding its customer personalities, enabling it to provide
better services and benefits to customers who have the potential to become loyal. By processing
historical marketing campaign data to improve performance and target the right customers to
transact on the company's platform, our focus is on creating a predictive clustering model to
assist the company in making decisions.

## Goal
The goal of analyzing customer profiles and behavior with a clustering approach is to understand customers better, provide more personalized service, improve sales performance, and build strong relationships with customers.

## Objective
Develop a machine learning model that can classify consumers into several groups according to their traits and actions.
Gain greater understanding of the behavior and profiles of your customers.
utilizing the results of clustering to identify profitable business strategies.

# Data Preparation
## Handling Data
- Handling missing values
- Handling Duplicated Data
- Handling the type and consistency of values
- Handling for outliers or unusual data (anomalies)

## Feature Engineering 
The explanation of each new features:

1.   **Age**: Provides demographic insight into customer age, influencing purchasing behaviors and marketing strategies.
2.   **AgeSegment**: Categorizes customers into 'Young Adults', 'Middle-aged Adults', and 'Old Adults' for targeted marketing and behavior analysis.
3.   **HasParent**: Indicates if a customer is a parent, crucial for understanding family-related purchasing decisions.
4.   **NumChildren**: Quantifies the number of children each customer has, influencing household spending patterns.
5.   **TotalAccCmp**: Counts accepted marketing campaigns, reflecting customer engagement and responsiveness.
6.   **TotalSpent**:  Sum of customer spending across product categories, indicating overall customer value.
7.   **TotalPurchases**: Total transactions made by each customer, showing purchasing frequency and loyalty.
8.   **ConversionRate**: Ratio of web purchases to web visits, measuring online sales effectiveness and customer engagement.


**For the detail kindly to check notebook**


# Data Exploration


![ScatterPlot](https://github.com/Rikaelisabeth09/Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb/blob/main/Scatterplot.png)

Insight: 

Overall, the data reveals several important relationships that can inform the business strategy:

1. Income and Conversion Rate: There is a positive correlation between customer income and conversion rate, though it is not a linear relationship. Identifying the optimal income range that maximizes conversion rates can help target the most valuable customer segments.
2. Spending and Conversion Rate: Total customer spending is strongly positively correlated with conversion rate. Customers who spend more tend to have higher conversion rates. Focusing on high-spending customers could be an effective strategy to drive overall conversion.
3. Age and Conversion Rate: Conversion rates exhibit a U-shaped relationship with customer age. Younger and older customers tend to have higher conversion rates compared to middle-aged customers. Understanding the needs and behaviors of different age groups can enable more tailored marketing approaches.
4. Income and Spending: There is a clear linear relationship between customer income and total spending. Higher-income customers spend more overall, which is valuable for segmentation and targeting.

These insights suggest that customer income, total spending, and age are important factors that influence the conversion rate for this business. Understanding these relationships can help optimize marketing strategies, product offerings, and customer engagement to improve overall conversion performance.




![Correlation Heatmap](https://github.com/Rikaelisabeth09/Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb/blob/main/Heatmap.png)

Insight :  
From the correlation heatmap, the highest correlation is between Income and Conversion Rate, which is 0.52. This positive correlation suggests that the higher the customer's income, the higher their conversion rate tends to be.

This can be clearly seen in the "Conversion Rate Analysis Based on Income" scatter plot, where the data points show an upward trend - the higher the income, the higher the corresponding conversion rate.

So in summary, the correlation heatmap and the related scatter plot indicate a strong positive relationship between customer income and conversion rate, where higher income customers have higher conversion rates.


# Data Modeling with K means 

## Data Preprocessing 
Before proceeding with data modeling, several key pre-processing steps must be completed:

- Remove Unnecessary Features: Eliminate features that are not relevant to the model. This helps streamline the dataset and ensures that the focus remains on the most pertinent information.

- Encode Categorical Features: Transform categorical features into numerical values through encoding techniques. This step is crucial as machine learning algorithms typically require numerical input to process the data effectively.

- Standardize Features: Apply feature standardization to normalize the scale of the data. This step ensures that all features contribute equally to the model, preventing any single feature from disproportionately influencing the outcome due to its scale.



## Modeling 
Using the Principal Component Analysis (PCA) approach comes next after data pre-processing. PCA is used to keep important information while reducing the dimensionality of the data.

![PCA](https://github.com/Rikaelisabeth09/Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb/blob/main/PCA%201.png)

![PCA](https://github.com/Rikaelisabeth09/Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb/blob/main/PCA%202.png)

### Insight : 
Based on the information provided in the images, here is a summary of the relationship between the number of clusters and the silhouette score as well as the inertia score:
1. Silhouette Score:
    As the number of clusters increases, the silhouette score generally decreases. This is because with more clusters, it becomes more difficult to find a clear separation between the clusters, leading to lower silhouette scores. The silhouette score is often used to determine the optimal number of clusters, as a higher score indicates better cluster separation and assignment.

2. Inertia Score:
    As the number of clusters increases, the inertia score generally decreases. This is because with more clusters, the data points are more tightly grouped within each cluster, leading to a lower inertia score. The inertia score is often used in conjunction with the silhouette score to determine the optimal number of clusters, as a balance between cluster separation and compactness is desirable.

The optimal number of clusters is typically found by examining the "elbow" or "knee" in the plots of the silhouette score and inertia score against the number of clusters. This point often represents a good trade-off between the two metrics and can indicate the appropriate number of clusters for the dataset.


### Cluster Segmentation
![Cluster Segmentation](https://github.com/Rikaelisabeth09/Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb/blob/main/pca%20component.png)

### Insight 
The K-Means clustering analysis on the PCA components has revealed a well-structured partitioning of the data into 4 distinct clusters. The clusters exhibit clear separation and differences in their compactness, suggesting the algorithm has effectively captured the inherent groupings within the dataset. This analysis offers valuable insights into the underlying patterns and relationships in the data, providing a solid foundation for further investigation and exploration.


##  **Customer Personality Analysis**
The objective of customer personality analysis is to **identify distinctive qualities that each group may possess, as well as comprehend the contrasts and similarities between these clusters**. Businesses are better able to target more particular business strategies and take more appropriate action for each consumer group when they have a deeper understanding of the traits that separate the clusters.

![Customer Personality Analysis](https://github.com/Rikaelisabeth09/Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb/blob/main/number%20of%20cluster.png)


## Interpretation Cluster : 

Cluster Interpretation and Marketing Recommendations

1. High Spender:
    
    The High Spender cluster is characterized by the highest sum total spending of $698,502,000.00, an average conversion rate of 2.17, and an average income of $75,700,175.30. This cluster consists of customers who spend the most on the platform, demonstrating a high conversion rate and substantial purchases during their infrequent visits. Their high income supports their spending behavior, focusing on premium products.
2. Mid Spender:
    
    The Mid Spender cluster has a sum total spending of $502,205,000.00, an average conversion rate of 1.33, and an average income of $62,060,108.66. These customers have the highest average total transaction of 22.43 and, like the high spenders, focus on premium products. This group shows a moderate level of spending and income, with a good conversion rate. Their average total transaction is higher than that of the High Spender cluster, indicating frequent and significant purchases.
3. Low Spender:
    
    The Low Spender cluster has a sum total spending of $42,039,000.00, an average conversion rate of 0.30, and an average income of $29,859,652.72. These customers have lower conversion rates and spending levels compared to other clusters, with an average total transaction of 6.73. This cluster represents customers with lower conversion rates and spending levels. Despite their relatively high income, their spending behavior suggests potential for increased engagement.
4. Risk Churn:
    
    The Risk Churn cluster is characterized by the lowest sum total spending of $102,901,000.00, the lowest average conversion rate of 0.48, and an average income of $41,346,600.46. These customers often visit the platform but do not complete transactions. They primarily spend on Coke, meat, and gold products. Customers in this cluster have low conversion rates and spending, with a potential risk of churn. They often visit the platform without completing transactions, indicating a need for retention strategies.
    
##  **Business Recommendation**
1. High Spender:
   Marketing Recommendation:
    *   Personalize retargeting campaigns based on previous purchases.
    *   Emphasize exclusive offers on premium products.
    *   Utilize high-income targeting for promotions and loyalty programs to increase spending.

2. Mid Spender:
   Marketing Recommendation:
    *   Implement retargeting strategies to promote a broader range of products.
    *   Introduce loyalty programs to encourage more frequent visits and higher spending.
3. Low Spender:
   Marketing Recommendation:
    *   Implement targeted promotions for a wider range of products.
    *   Launch special offers and discounts to attract this segment and increase their frequency of visits.
4. Risk Churn:
    Marketing Recommendation:
    To prevent churn and increase re-engagement, it is recommended to implement aggressive retargeting campaigns         with personalized incentives. Focus on customer satisfaction initiatives and exclusive offers are also crucial       to win back and retain these customers.
