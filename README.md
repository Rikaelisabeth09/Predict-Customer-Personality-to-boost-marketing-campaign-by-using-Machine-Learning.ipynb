# Predict-Customer-Personality-to-boost-marketing-campaign-by-using-Machine-Learning.ipynb
This repository contains the code and analysis for predicting customer personality traits to enhance marketing campaigns using machine learning techniques. The project aims to segment customers based on their behavior and characteristics to optimize marketing strategies and improve campaign performance.

**Table of Contents**
- [Problem Statement](#problem-statement)
    - [Project Overview](#projectoverview)
    - [Goal](#goal)
    - [Objective](#objective)
- [Data Preparation](#data-preparation)
    - [Handling Data](#stage-1-data-preparation)
    - [Feature Engineering](#feature-engineering)
- [Data Exploration](#data-exploration)
- [Data Modeling K means](#data-modeling-with-k-means)
    - [Preprocessing](#stage-3-data-modeling-with-k-means)
    - [Modeling](#modeling)
    - [Cluster Segmentation](#cluster-segmentation)
- [Customer Personality Analysis](#customer-personality-analysis)
- [Business Recomendation](#stage-5-business-recommendation)


# Problem Statement

## Background
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
<br>
<br>


# Data Modeling with K means 
## Data Preprocessing 
Before proceeding with data modeling, several key pre-processing steps must be completed:

- Remove Unnecessary Features: Eliminate features that are not relevant to the model. This helps streamline the dataset and ensures that the focus remains on the most pertinent information.

- Encode Categorical Features: Transform categorical features into numerical values through encoding techniques. This step is crucial as machine learning algorithms typically require numerical input to process the data effectively.

- Standardize Features: Apply feature standardization to normalize the scale of the data. This step ensures that all features contribute equally to the model, preventing any single feature from disproportionately influencing the outcome due to its scale.



## Modeling 
Using the Principal Component Analysis (PCA) approach comes next after data pre-processing. PCA is used to keep important information while reducing the dimensionality of the data. It can improve model performance and solve the issue of multicollinearity between features by lowering the dimensionality of the data. Finding the ideal number of clusters is a crucial next step in this process. The Elbow Method and the Distortion Score are employed in this research to determine the ideal number of clusters. The analysis's findings indicated that four clusters was the ideal quantity.

<p align="center">
    <kbd> <img width="800" alt="elbow method" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/f566868f-9f86-4288-9455-47c5ca2144c8"></kbd> <br>
    Figure 3: Elbow Method
</p>


<br>

<p align="center">
    <kbd><img width="600" alt="hotel resort" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/a783fcc7-7237-45db-be81-c8b0ef3361ad"></kbd> <br>
    Figure 4 - Silhouette Score
</p>

### Result : 
Base on elbow method and silhoute score best cluster is 3.

### Cluster Segmentation
<p align="center">
    <kbd><img width="1000" alt="segment" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/a2425656-406e-4682-91a9-30c70b27afa4"></kbd> <br>
    Figure 5 - Scatter Plot Cluster segmentation
</p>

### Result 
From the results of the scatterplot above, it can be said that the number of clusters equal to 3 is the right number of clusters. where it can be seen that there is a fairly clear segmentation between the clusters.
<br>


##  **STAGE 4: Customer Personality Analysis**
The objective of customer personality analysis is to **identify distinctive qualities that each group may possess, as well as comprehend the contrasts and similarities between these clusters**. Businesses are better able to target more particular business strategies and take more appropriate action for each consumer group when they have a deeper understanding of the traits that separate the clusters.

<p align="center">
    <kbd> <img width="700" alt="3d visual" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/07620944-ae77-4e00-89d7-fe65093b37db"></kbd> <br>
    Figure 6 — 3d visual cluster segmentation
</p>


## Interpretation Cluster : 

- cluster 0 
    - cluster 0 is a group of customers who frequently carry out transactions 21 times and have spent Rp. 1.169.791/month
    - According to other groups, this cluster visits the website an average of 3 times, but their conversion rate is rather high. This suggests that this particular consumer type only visits the website when they are certain they want to buy a product.
    - The majority of these customers are 56 years old
    - With a monthly income of Rp. 71.250,297 so it is not surprising that this cluster spends more than the other clusters.
    - This customer represents `can't loose them`
- cluster 1 
    - This cluster only spends IDR. 74,116 and only made 8 transactions and max 18 transaction.
    - Despite frequently visiting the website, this cluster has the lowest conversion rate, showing that the customers in consideration are only typical product browsers and have no plans of completing a purchase.
    - this group only spent Rp. 91.391 , this is very low spent rather than other cluster.
    - this cluster just have 1% conversion rate.
    - This cluster represents `low customers`
- cluster 2 
    - This cluster carried out 17 transactions and spent Rp. 639.300
    - This customer group also has an average income of around Rp. 55.828.696
    - Compared to the cluster 0 group, this customer group sees the website more frequently and has a decent conversion rate of about 4%.
    - If we can provide this cluster with the proper discount, it could be able to boost its conversion rate.
    - This customer represents a `potential customer`
- **For the detail insight kindly to check notebook**
<br>

The characteristics of each group, the cluster's propensity to react to current marketing campaigns, and the possible revenue results from marketing retargeting to the cluster can all be used to interpret the results of prior clustering. Additional analysis will be performed.

<p align="center">
    <kbd> <img width="800" alt="percentage" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/1fb7e721-3320-4ac3-afd3-6c1a8d579330"></kbd> <br>
    Figure 7 — Percentage total customer and average spent by cluster
</p>

### Result : 
- The graph indicates that the cluster with the lowest group has the greatest percentage of customers (54%), but the average amount spent is quite low. To boost the amount spent for this cluster group, consider offering exclusive discounts to draw in cluster groups with the greatest number of customers. 
- It is evident to prospective buyers that 28% of this group's consumers have the capacity to develop into loyal customers with the proper treatment, one way to demonstrate this is by giving clients a positive shopping experience and showing gratitude through a loyalty program.
- Even though the customer group can't loose them, it has a small number of customers, but it has the highest average spent compared to other clusters.
<br>

Analysis of the distribution of several features of each cluster was also carried out to gain deeper insight. Through this analysis, several interesting insights were discovered that can provide a better understanding of user behavior in each cluster, especially regarding website visits, conversion rate and Total Spent.

<p align="center">
    <kbd> <img width="700" alt="percentage" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/0c7633e4-252c-42b2-ba2e-e7f4f207399d"></kbd> <br>
    Figure 8 — Distribution plot by cluster
</p>

### Interesting thing : 
- low customer have the lowest income, Maybe that's why the conversion rate is low.
- The three customer groups have almost the same distribution of recency.
- According to the interpretation above, website visits for low customers are the highest.

In order to maximize the business metric `GMV`, we will perform a multivariate analysis with Using simple statistical regression , to determine which features have the greatest influence with total spent feature.

<p align="center">
    <kbd> <img width="800" alt="percentage" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/5bcff300-51bf-4502-bfbf-406f04694dca"></kbd> <br>
    Figure 9 — Regplot by cluster
</p>

### Result : 
- Features such as `income`, `total_spent`, `conversion_rate` these three features have a very strong influence on spending made by customers, therefore the company can provide appropriate recommendations for these features in order to increase `GMV` according to the target .
- Another interesting finding is that features like `visit website` have a negative correlation, indicating that a significant number of users are still viewing the website but are not completing any purchases. As a result, it is imperative to evaluate the current website in order to draw in more visitors. individually to attract their interest in completing a purchase.

<br>
<br>

##  **STAGE 5: Business Recommendation**

### Potential Impact :
Total Spent Group cant' loose them: Rp. 397.729.000<br>
Total Spent Group low customer: Rp. 90.751.000<br>
Total Spent Group potensial customer: Rp. 324.125.000<br>
Total all cluster : Rp. 812,605.000

We still have a potential gross merchandise value (`GMV`) of IDR 812,605,000 million if we can continue to prioritize our present customers and no one leaves.

## Recomendation

1.  `Can't Loose Them` <br>
The advice that can be given is to offer loyalty programs, such as free shipping discounts and special promotions for customer groups, because this customer group frequently makes transactions and the total amount spent is fairly high. This will help them be more satisfied with their purchases and keep their interest in the group.

2. `Low Customer` <br>
customers have the fewest spending and transactions of any group, but they have one unique characteristic they visit the website frequently so it can be advised to do an evaluation on the website to draw in this particular customer group. If needed, customer groups can also be offered small discounts to encourage more transactions, which will minimize the possibility of churn in this customer group.

3. `Potential Customer` <br>
If the company treats this customer group well, it could become the first cluster group. For example, by sending relevant and engaging messages to customers, it can guarantee a positive user experience when they visit the website or interact with the company's offerings. Developing a loyalty or incentives program helps improve client interaction. Companies may motivate potential loyal customers to keep selecting and buying their goods and services by offering them incentives like points, rewards, or exclusive advantages.
