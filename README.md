## Customer-Segmentation-Project

## Introduction
Customer segmentation is the process of dividing customers into groups based on common characteristics so companies can market to each group effectively and appropriately.

In business-to-business marketing, a company might segment customers according to a wide range of factors, including:
* Industry
* Number of employees
* Products previously purchased from the company
* Location

In business-to-consumer marketing, companies often segment customers according to demographics that include:

* Age
* Gender
* Marital Status
* Location (urban, surburban, rural)


## Objective

The aim of this project is to determine the types of customers (target customers) who can easily convert into loyal customers so that the marketing team can make an informed decision about their approach. 

## Stakeholders

The results obtained from this project can be used by various stakeholders within the bank such as
* Marketing team
* Sales team
* Customer success team

## Importance of the project

Segmentation allows marketers to better tailor their marketing efforts to various audience subsets. Those efforts can relate to both communications and product development. Specifically, segmentation helps a company:
* Create and communicate targeted marketing messages that will resonate with specific groups of customers, but not with others (who will receive messages tailored to their needs and interests, instead).
* Select the best communication channel for the segment, which might be email, social media posts, radio advertising, or another approach, depending on the segment. 
* Focus on the most profitable customers.
* Improve customer service.

## Code and Resources used

**Python Version**:3.9.12 

**Packages**:Pandas,Numpy,Scikit learn,Matplotlib,Seaborn,Imblearn, Collection, Intertools

**Data Source**:https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

## Data Collection
The datasets used in this project were downloaded from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. I then read the two csv files using the pd.read_csv() command.

## Data Cleaning
After downloading the data, I needed to clean it up so that it was usable for our model. In our case, the dataset did not contain any missing values and the data was of the correct format.

## Exploratory Data Analysis (EDA)
The data only consists of 4 numerical variables, and 1 categorical variable. I looked at different distributions for both the categorical and numeric data. Below are highlights from the data visualization section

![bar graph](https://github.com/MusaMasango/Credit-Card-Fraud-Detection-Project/blob/main/labels%20bar%20graph.png)
![correlation matrix](https://github.com/MusaMasango/Credit-Card-Fraud-Detection-Project/blob/main/correlation%20matrix.png)

## Model Building 

I will first implement the kmeans algorithm on the data set having 200 rows and 5 columns dataset and see how it works. The data contains the age of the customers, annual income, spending score as well as customer ID. I will be looking at the different relationships between these attributes :
* Age and Spending Score
* Annual Income and Spending Score
* Age and Annual Score

From now on we will be using sklearn implementation of kmeans. Few thing to note here:

* n_init is the number of times of running the kmeans with different centroid’s initialization. The result of the best one will be reported.
* tol is the within-cluster variation metric used to declare convergence.
* The default of init is k-means++ which is supposed to yield a better results than just random initialization of centroids.

Next, we'll calculate the sum of squared error as well as the silhouetter values using different number of clusters then plot the results.

## Model Performance
Contrary to supervised learning where we have the ground truth to evaluate the model’s performance, clustering analysis doesn’t have a solid evaluation metric that we can use to evaluate the outcome of different clustering algorithms. Moreover, since kmeans requires k as an input and doesn’t learn it from data, there is no right answer in terms of the number of clusters that we should have in any problem.

In order to evaluate the model performance for the different the different clusters, two clustering metrics were used:

* Elbow method
* Silhouette analysis

Elbow Method

Elbow method gives us an idea on what a good k number of clusters would be based on the sum of squared distance (SSE) between data points and their assigned clusters’ centroids. We pick k at the spot where SSE starts to flatten out and forming an elbow. We’ll use the geyser dataset and evaluate SSE for different values of k and see where the curve might form an elbow and flatten out. Below are the plots obtained using this method

Silhouette analysis

Silhouette analysis can be used to determine the degree of separation between clusters. For each sample:

* Compute the average distance from all data points in the same cluster (ai).
* Compute the average distance from all data points in the closest cluster (bi).
* Compute the coefficient:
    equation
The coefficient can take values in the interval [-1, 1].
* If it is 0 –> the sample is very close to the neighboring clusters.
* It it is 1 –> the sample is far away from the neighboring clusters.
* It it is -1 –> the sample is assigned to the wrong clusters.

Therefore, we want the coefficients to be as big as possible and close to 1 to have a good clusters.

## Conclusion
1. In this python machine learning project, I built a binary classifier using the 6 algorithms to detect credit card fraud transactions. Through this project, I applied techniques to address the class imbalance issues and achieved an accuracy of more than 90%. The random forest model yields a very good performance as indicated by the model accuracy which was found to be 0.99990035.
2. To address the issue of class imbalance problem, we used the oversampling technique, this was done by the SMOTE package imported from the imblearn module.
3. ROC AUC of our models approaches towards 1. So, we can conclude that our classifier does a very good job in predicting whether a transcation is genuine or fraud. 
