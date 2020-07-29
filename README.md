# Abalone_Age_prediction

The two notebooks are implemented in GOOGLE COLAB(a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud.)

Implemented in Notebook age_of_an_abalone.ipynb

###### Task- *Predict the age of abalone from physical measurements only.*

###### Problem- *Typically, the age of an abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope. This process is tedious and time-consuming.*

###### Solution- *Create a machine learning algorithm that will predict the age of an abalone from physical measurements only.*


## Dataset information
* 8 attributes
* 4177 instances

1.Sex: nominal - M (male), F (female), I (infant)

2.Length: continuous - longest shell measurement (in mm)

3.Diameter: continuous - measurement perpendicular to legnth (in mm)

4.Height: continuous - with meat in shell (in mm)

5.Whole weight: continuous - the whole abalone (in grams)

6.Shucked weight: continuous - weight of the meat (in grams)

7.Viscera weight: continuous - gut weight after bleeding (in grams)

8.Shell weight: continuous - after being dried (in grams)

9.Rings: integer - +1.5 gives the age in years

###### The notebook age_of_an_abalone.ipynb has the details of the implementation.The steps involved are:
1. Import data- using pd.read_csv function to load data fromcsv file 

2. Overview of data- Using profilereport function frompandas profiling we get get overall statistic of data variables and the correlations between the features.

3. Exploratory Data Analysis (EDA) along with one-hot encoding.

4. Modelling- the sub parts implemented in modelling part :

   * Split dataset
   * Baseline model: multiple linear regression , 
   * Multiple linear regression with regularization (ridge regression)
   * Multiple linear regression with regularization (lasso)
   * Decision-tree based models: Random forest, Bagging, LightGBM
   * Compiling the results
 
 
The RMSE error was least for LightGBM

###### The results are:

Model and RMSE error

 1.linear regression - 1.5662            
 2.linear regression + ridge - 1.5641                        
 3.linear regression + lasso -	1.5564                         
 4.random forest - 2.2334                            
 5.Bagging	- 2.2334                                  
 6.LightGBM - 	1.5097

# Credit Card Fraud Detection

It is a [Kaggle challenge](https://www.kaggle.com/mlg-ulb/creditcardfraud)

###### Task- *recognize fraudulent credit card transactions in highly unbalanced datset.*
###### Content-
*This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.The positive class (frauds) account for 0.172%
of all transactions.*           

*It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data.*               

*Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.*

> We will use various predictive models to see how accurate they are in detecting whether a transaction is a normal payment or a fraud. As described in the dataset, the features are scaled and the names of the features are not shown due to privacy reasons. Nevertheless, we can still analyze some important aspects of the dataset.

The Credit_Card_Fraud_Detection.ipynb contains 3 various models to solve the task.
The first task is implemented based on the main sub goals which are:

* Understand the insignificant distribution of the "little" data that was provided to us.
* Create a 1:1 sub-dataframe ratio of "Fraud" and "Non-Fraud" transactions. (NearMiss Algorithm)
* Determine the Classifiers we are going to use and decide which one has a higher accuracy.
* Create a Neural Network and compare the accuracy to our best classifier.
* Analyse common mistakes made with imbalanced datasets.

###### Outline:         
I. Insight on our data:
a) Gather Sense of our data

II. Preprocessing:                         
a) Scaling and allocating
b) Splitting the Data


III. Random UnderSampling and Oversampling:      
a) Distributing and Correlating
b) Anomaly Detection
c) Dimensionality Reduction and Clustering (t-SNE)
d) Classifiers
e) Examining into Logistic Regression
f) Oversampling with SMOTE


IV. Testing                                       
a) Testing with Logistic Regression
b) Neural Networks Testing (Undersampling vs Oversampling)

The result conclusion is:                   
1)Random UnderSampling	- 0.942105                             
2)Oversampling (SMOTE)	- 0.987079  

Classification Models: The models that performed the best were logistic regression and support vector classifier (SVM)

***References:***                                              
[Kaggle notebook By janio martinez](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)               
[DEALING WITH IMBALANCED DATA](https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation)

The second method used is by anamoly detection Techniques namely:
* Isolation Forest Anomaly Detection Algorithm
* Density-Based Anomaly Detection (Local Outlier Factor)Algorithm
* Support Vector Machine Anomaly Detection Algorithm

Observations basedonthe results obtained in this model are:                  
* Isolation Forest has a 99.74% more accurate than LOF of 99.65% and SVM of 70.09
* When comparing error precision & recall for 3 models , the Isolation Forest performed much better than the LOF as we can see that the detection of fraud cases is around 27 % versus LOF detection rate of just 2 % and SVM of 0%.
* So overall Isolation Forest Method performed much better in determining the fraud cases which is around 30%.

***References:***                                                 
[Expanation by Pavan sangapati](https://www.kaggle.com/pavansanagapati/anomaly-detection-credit-card-fraud-analysis)

The third method used is using AutoEncoders (Semi Supervised Classification)              

The steps are:                            
* Visualize Fraud Vs Non Fraud Transactions
* Obtain the Latent Representations by AutoEncoders
* Simple Linear Classifier based on the Visualizing Latent Representations of Fraud vs Non Fraud

The accuracy score obtained was 0.98.

***References:***                                                          
[Shivam Bansal's Blog](https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders)



