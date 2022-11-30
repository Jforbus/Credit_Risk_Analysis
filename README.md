# Credit_Risk_Analysis

## Project Overview
Using the `Imbalanced-learn` library and `scikit-learn` to build and evaluate several models for use in assessing credit risk. Several methods are assessed for their ability to handle the inherent bias of an excessively small target data set. Data from LendingClub's Q1-2019 loan data is used to train and test the models. The project compares the viablilty of 4 resampling methods paired with `LogisticRegression`: `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, and `SMOTEEN`. Next, the data is used to train and assess two ensemble classifiers: `BalancedRandomForest` and `EasyEnsembleClassifier`.

### Resources
- Data Source: [LoanStats_2019Q1.csv](https://raw.githubusercontent.com/Jforbus/Credit_Risk_Analysis/main/Resources/LoanStats_2019Q1.csv)
- Software: Python 3.7.15, scikit-learn 1.0.1, imbalanced-learn 0.9.0 pandas 1.3.5, numpy 1.21.5, Jupyter Notebook

## Results
An accuracy report, confusion matrix, and classification report is generated to compare each of the models based on accuracy, precision, and recall.

---

### **Resampling Methods**

**RandomOverSampler and LogisticRegression**


![ros_bar](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/ros_bar.png "Balanced Accuracy Report")
![ros_confmx](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/ros_confmx.png "Confusion Matrix")
![ros_class](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/ros_class.png "Classification Report")


- With the data resampled using RandomOverSampler the Logistic Regression Model has an accuracy score of: 0.64
- The model determines high_risk cases with a Precision of: 0.01
- The model determines high_risk cases with a Recall of: 0.69
- This model appears have decent accuracy due to the extreme imbalance in the data, but its moderate sensitivity and extremely low precision results in a relatively large number of false identifications of high_risk cases and a relatively large number unidentified high_risk cases.



**SMOTE and LogisticRegression**


![smote_bar](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/smote_bar.png "Balanced Accuracy Report")
![smote_confmx](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/smote_confmx.png "Confusion Matrix")
![smote_class](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/smote_class.png "Classification Report")


- With the data resampled using SMOTE the Logistic Regression Model has an accuracy score of: 0.66
- The model determines high_risk cases with a Precision of: 0.01
- The model determines high_risk cases with a Recall of: 0.63
- Again, this model appears have decent accuracy due to the extreme imbalance in the data, but its moderate sensitivity and extremely low precision results in a relatively large number of false identifications of high_risk cases, though less than the previous model, as well as a large number of unidentified high_risk cases.



**ClusterCentroids and LogisticRegression**


![cc_bar](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/cc_bar.png "Balanced Accuracy Report")
![cc_confmx](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/cc_confmx.png "Confusion Matrix")
![cc_class](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/cc_class.png "Classification Report")


- With the data resampled using ClusterCentroids the Logistic Regression Model has an accuracy score of: 0.54
- The model determines high_risk cases with a Precision of: 0.01
- The model determines high_risk cases with a Recall of: 0.69
- This model appears to be less accurate than the previous two, and suffers from the same lack of precision. Its high sensitivity and extremely low precision results in an even larger number of false identifications of high_risk cases, as well as a significant number of unidentified high_risk cases.



**SMOTEEN and LogisticRegression**


![combo_bar](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/combo_bar.png "Balanced Accuracy Report")
![combo_confmx](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/combo_confmx.png "Confusion Matrix")
![combo_class](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/combo_class.png "Classification Report")


- With the data resampled using SMOTEEN the Logistic Regression Model has an accuracy score of: 0.67
- The model determines high_risk cases with a Precision of: 0.01
- The model determines high_risk cases with a Recall of: 0.76
- The results resemble the previous methods, with a moderate recall and complete lack of precision generating a large number of falsely identified high_risk cases, and a non-negligible number of unidentified high_risk cases.

---

### **Ensemble Classifiers**

**BalancedRandomForest**


![brf_bar](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/brf_bar.png "Balanced Accuracy Report")
![brf_confmx](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/brf_confmx.png "Confusion Matrix")
![brf_class](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/brf_class.png "Classification Report")


- The BalancedRandomForest Model has an accuracy score of: 0.79
- The model determines high_risk cases with a Precision of: 0.03
- The model determines high_risk cases with a Recall of: 0.70
- This model presents an incremental increase in precision over the previous models while maintaining high recall, significantly lowering the number of falsely identified high_risk cases. This model , like those before, failed to identify a significant number of high_risk cases. 



**EasyEnsembleClassifier**


![eec_bar](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/eec_bar.png "Balanced Accuracy Report")
![eec_confmx](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/eec_confmx.png "Confusion Matrix")
![eec_class](https://github.com/Jforbus/Credit_Risk_Analysis/blob/main/Resources/eec_class.png "Classification Report")


- The EasyEnsembleClassifier Model has an accuracy score of: 0.93
- The model determines high_risk cases with a Precision of: 0.09
- The model determines high_risk cases with a Recall of: 0.92
- With the highest accuracy, precision, and sensitivity, the EasyEnsembleClassifier performed best with this data. Nearly all high_risk cases were positively identified by the model, and the number of false identifications has been decreased significantly.



## Summary

Identifying credit risk is inherently tricky given the significant imbalance in the target pool. High risk cases accounted for about .5% of the data used. Training a model to correctly identify high risk cases requires resampling to correct for this imbalance. The first four methods tried: `RandomOverSampler`, `SMOTE`, `ClusterCentroids`, and `SMOTEEN`, are completely unreliable. Not only do they fail to correctly identify high risk cases at an acceptable rate, they also generate thousands of false identifications. With at least 25% of the high risk cases in the testing data missed, these models would do more harm than good. 
The Ensemble Classifiers did significantly better, one moreso than the other. `BalancedRandomForest` was able to increase precision and eliminate a lot of false positives, but created a significant number. Perhaps more importantly it continued the trend of missing a significant number of high risk cases, with 30% wrongly categorized as low risk. The model with the greatest performance is `EasyEnsembleClassifier`. This model had significantly better results than the others. Greater than 90% of the high risk cases were correctly identified, and less than 1000 false positives were generated. 

Of the models tested here `EasyEnsembleClassifier` is the clear choice for assessing credit risk given the results above. With more tweaking and modification the precision and recall may be increased to further decrease the number of false positives and missed cases. Further research should be conducted to determine if the precision and sensitivity of `EasyEnsembleClassifier` can be increased to an acceptable rate, or if other models or methods can better categorize the data.