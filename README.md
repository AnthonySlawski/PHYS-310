# PHYS-310
## Machine Learning for Physics and Astronomy Data Analysis 

This repository is dedicated to a collection of machine learning tasks and methodologies I learnt from my course PHYS310, covering both fundamental concepts and advanced techniques. It serves as a valuable resource for anyone interested in machine learning. Below is an outline of the key areas covered:

## [Classification Tasks](https://github.com/AnthonySlawski/PHYS-310/blob/main/Classification%20Tasks.ipynb)
- **Decision Trees:** Investigate the application of decision trees in classification assignments, a foundational method in machine learning that models decisions and their possible consequences.
- **K-Nearest-Neighbours (KNN):** Classify instances based on the closest training examples in the feature space.
- **Support Vector Machines (SVM):** Effectively perform classification by finding the optimal separating hyperplane.
- **Logistic Regression:** A fundamental statistical model used for binary classification tasks.
## [Regression Tasks](https://github.com/AnthonySlawski/PHYS-310/blob/main/Regression%20Tasks.ipynb)
- **Linear Models:** Predicting a quantitative response.
- **Regularization:** Using techniques like Lasso and Ridge regression to prevent overfitting and enhance model generalization.
- **Gradient Descent:** Crucial for learning the parameters of various models.
## [Evaluation and Optimization of ML Models](https://github.com/AnthonySlawski/PHYS-310/blob/main/Evaluation%20and%20Optimization%20of%20ML%20Models.ipynb)
- **Cross Validation:** Assessing the predictive performance of your models and ensuring they generalize well to unseen data.
- **Hyperparameter Optimization:** Optimizing the hyperparameters of your machine learning models to improve performance.
- **Feature Engineering:** Understanding the importance of feature selection and transformation in developing robust models.
## [Ensemble Methods](https://github.com/AnthonySlawski/PHYS-310/blob/main/Ensemble%20Methods.ipynb)
- **Random Forests:** An ensemble learning method for classification and regression that improves predictive accuracy.
- **Boosting Methods:** Combine multiple weak learners to form a strong learner, enhancing model predictions.
## [Dimensionality Reduction](https://github.com/AnthonySlawski/PHYS-310/blob/main/Dimensionality%20Reduction.ipynb)
- **Clustering:** Grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.
- **Principal Component Analysis (PCA):** A technique used to emphasize variation and bring out strong patterns in a dataset.
## [Neural Networks and Deep Learning](https://github.com/AnthonySlawski/PHYS-310/blob/main/Neural%20Networks%20and%20Deep%20Learning.ipynb)
- **Activation Functions:** Crucial for introducing non-linearity into the model.
- **Training and Backpropagation:** Fundamentals of training neural networks and the backpropagation algorithm for adjusting the weights.
- **Convolutional Nets:** Especially powerful in processing data with a grid-like topology, such as images. **Checking out the famous [MNIST dataset](https://www.engati.com/glossary/mnist-dataset#:~:text=The%20MNIST%20database%20(Modified%20National,the%20field%20of%20machine%20learning. )!**

This repository aims to provide an overview and practical examples of each of these areas, with a broad spectrum of machine learning tasks and techniques.

## Acknowledgments

Special thanks to Dr. Joerg Rottler for his invaluable guidance and support throughout the development of this repository. His knowledge and understanding have been crucial to the project's success.

# Let's get started
## [Lab Notebook 1: Classical Inference vs Machine Learning](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%201.ipynb)

**Objective:** This notebook aims to review basic concepts of linear regression and contrast them with a machine learning approach, highlighting the differences and similarities between classical statistical inference methods and more modern machine learning techniques.

**Methods Used:**

- Linear regression models are discussed and implemented to show how classical inference operates.
- Key Python libraries used include NumPy, SciPy, Matplotlib, Pandas, and StatsModels.
- The notebook begins by importing necessary modules and setting up parameters for figure aesthetics.

**Key Findings:**

- Initial steps involve loading essential modules (NumPy for numerical computations, Matplotlib and Matplotlib.pyplot for plotting, and additional libraries for statistical modeling).
- The approach starts with a fundamental statistical method to set the stage for contrasting with machine learning techniques later on.

**Conclusion:** The notebook serves as an introductory comparison between classical statistical methods and machine learning, starting with the foundational concept of linear regression. It is aimed at setting the context for understanding how traditional methods of inference contrast with machine learning approaches in data analysis.

## [Lab Notebook 2 - Classical Inference vs Machine Learning](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%202.ipynb)

**Objective:** Building on the concepts introduced in the first lab, this notebook shifts the focus from classical inference to a machine learning (ML) perspective using the same dataset for consistency. The aim is to illustrate how machine learning approaches can be applied to data analysis and how they differ from classical statistical methods.

**Methods Used:**

- Continuation with Python libraries like NumPy for data manipulation and Matplotlib for visualizations.
- The notebook starts by re-establishing the environment setup from Lab 1, including module imports and random seed initialization to ensure reproducibility.
- It involves generating a dataset with a predefined relationship and adding randomness to simulate real-world data.
**Key Findings:**
- Initial steps include reusing the dataset from Lab 1 but with an intention to apply machine learning techniques.
- Data is prepared with a specific mathematical relationship and a layer of randomness to mimic actual experimental data.
- The approach emphasizes understanding the dataset's structure and behavior before applying machine learning models.
**Conclusion:** Lab Notebook 2 serves as a bridge between classical statistical methods and machine learning, using a consistent dataset to highlight how data analysis can be approached differently. By maintaining the same data, it allows for a direct comparison of methodologies and showcases the flexibility and power of machine learning techniques in extracting insights from data.

## [Lab Notebook 3 - kNN Algorithm](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%203.ipynb)

**Objective:** This notebook focuses on the k-Nearest Neighbors (kNN) algorithm, a fundamental technique in machine learning. The objective is to implement the kNN algorithm from scratch and then compare its performance with a pre-existing routine from a machine learning library.

**Methods Used:**

- Essential Python libraries for data manipulation and visualization are employed, including NumPy, Pandas, and Matplotlib. For machine learning, the notebook utilizes the scikit-learn library.
- The notebook instructs on organizing data for machine learning, starting with importing datasets, preprocessing, and visualizing the data.
- It emphasizes the importance of standardizing data and understanding the dataset's structure before applying machine learning models.
**Key Findings:**

- Data from a CSV file is loaded into a Pandas DataFrame for analysis. This dataset appears to be related to the habitability of exoplanets, considering features like planetary mass, orbital period, distance, and a binary habitability status.
- Initial data preparation steps include visual inspection of the dataset, highlighting the significance of understanding the data's characteristics and distribution before modeling.
**Conclusion:** Lab Notebook 3 provides a hands-on introduction to implementing the kNN algorithm from scratch, showcasing the power of this simple yet effective machine learning technique. Through data preparation, implementation, and comparison with established libraries, the notebook illustrates the practical steps and considerations in applying kNN to a real dataset, focusing on the importance of preprocessing and standardizing data for machine learning tasks.

[Lab Notebook 4 - Decision Trees](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%204.ipynb)

**Objective:** This notebook delves into Decision Trees (DT), a fundamental machine learning algorithm for classification and regression tasks. The exercise involves coding parts of a Decision Tree algorithm from scratch and comparing the results with a pre-written DT routine from a machine learning library.

**Methods Used:**

- Utilizes Python libraries such as NumPy for numerical operations, Pandas for data manipulation, and Matplotlib for visualization. For machine learning, it employs scikit-learn's DecisionTreeClassifier, among other utilities.
- The data preparation process mirrors that of the previous lab notebook, emphasizing the importance of loading, cleaning, and understanding the dataset before model implementation.
- The notebook guides through the implementation of a basic Decision Tree algorithm, highlighting key steps such as feature selection, tree construction, and node splitting.
**Key Findings:**

- The same dataset from the previous lab notebook is reused, ensuring continuity and providing a basis for comparing different machine learning algorithms' performance on a consistent dataset.
- Initial steps include setting up the environment, importing the necessary libraries, and preparing the dataset for analysis.
**Conclusion:** Lab Notebook 4 offers a comprehensive introduction to Decision Trees, showcasing the algorithm's logic and functionality through a hands-on coding exercise. By comparing a custom implementation with a pre-existing routine, the notebook provides insights into the algorithm's inner workings and the ease with which complex machine learning tasks can be accomplished using established libraries.

[Lab Notebooks 5 & 6 - Advanced Machine Learning Techniques](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%205-6.ipynb)
**Objectives:**
This notebook aims to apply k-nearest neighbors (kNN) and decision tree (DT) algorithms on a more extensive exoplanet dataset, exploring the effectiveness of the kNN model through various metrics and diagnostics.

**Methods Used:**

- Data Exploration: Utilization of pandas for data manipulation, matplotlib for graphing, and initial data exploration techniques to familiarize with the dataset's features and summary statistics.
- Model Application: Application of kNN and DT algorithms to classify exoplanet data, with an emphasis on model evaluation and interpretation of results.
**Key Findings:**

-Initial exploration provided insights into the dataset's structure, including the range and distribution of values across different features.
-Preliminary summary statistics offered a baseline understanding of the data's characteristics, laying the groundwork for more nuanced analysis through machine learning models.
**Conclusion:**
The lab underscores the importance of thorough data exploration as a precursor to model application. It demonstrates the potential of kNN and DT algorithms in classifying complex datasets, highlighting the need for careful selection and evaluation of machine learning models based on specific dataset characteristics.

[Lab Notebook 7: Support Vector Machine (SVM)](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%207.ipynb)
**Objectives:**
Introduction to the Support Vector Machine (SVM) algorithm, focusing on its application, visualization, and the impact of parameter tuning on the model's performance.

**Methods Used:**

- SVM Application: Usage of sklearn's svm.SVC with a linear kernel to fit the model to the dataset, experimenting with different values of the regularization parameter C to understand its effect on model complexity and decision boundaries.
- Visualization and Evaluation: Graphical representation of the SVM decision boundaries for different C values, aiding in the visualization of model behavior in relation to overfitting or underfitting.
**Key Findings:**

- The choice of C has a significant impact on the model's decision boundaries, reflecting a trade-off between model simplicity and fitting to the dataset.
- Visualization techniques serve as a powerful tool for understanding and interpreting model behavior, especially in relation to parameter choices.
**Conclusion:**
Lab Notebook 7 **elucidates** the foundational concepts of SVM, emphasizing the critical role of parameter tuning in optimizing model performance. It showcases SVM's applicability to classification problems and highlights the need for a balanced approach to model complexity.

[Lab Notebook 8: SVM Classification in Particle Physics](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%208.ipynb)
**Objectives:**
To explore the application of Support Vector Machines (SVMs) to a larger dataset from particle physics, focusing on the conversion of categorical labels into a binary format and handling missing data.

**Methods Used:**

- Data Preprocessing: Reading and exploration of the dataset using pandas, transformation of string-type labels into binary format with sklearn's LabelEncoder, and handling of missing data by considering a subset of the dataset with complete columns.
- SVM Application: Application of SVM to the processed dataset, with the aim of classifying particle physics events.
**Key Findings:**

- Preliminary data exploration revealed challenges associated with missing data and the need for careful preprocessing to ensure model applicability.
- Converting categorical labels into a binary format allowed for the straightforward application of SVM, demonstrating the algorithm's flexibility in handling various data types.
**Conclusion:**
This lab highlights the complexities of applying machine learning algorithms to real-world datasets, particularly in fields like particle physics where data can be incomplete or irregularly structured. It showcases the necessity of comprehensive data preprocessing and the potential of SVM in classifying complex datasets. ​​

[Lab Notebook 9: Basic Linear Regression](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%209.ipynb)
Objectives:
Explore the basics of linear regression, including setting up a model, exploring different loss functions, and understanding the concept of residuals in relation to model fit.

Methods Used:

Linear Model Setup: Utilized sklearn's LinearRegression() to fit a linear model to generated data, comparing the model's predictions against a predefined true regression line.
Residual Analysis: Investigated the residuals - the differences between the actual data points and the model's predictions - to assess the independence of errors from the independent variable (x).
Key Findings:

Residual analysis revealed that the residuals were not independent of x, indicating that the assumptions of the probabilistic linear model were not fully satisfied. However, it was still possible to create a predictive model despite this limitation.
Conclusion:
The lab demonstrates that while linear regression can provide a useful model for data prediction, understanding the nature of residuals and their relation to the independent variables is crucial for assessing model appropriateness and limitations.

Lab Notebook 10: Linear Regression with Gradient Descent
Objectives:
Introduce gradient descent as a method for finding the parameters of a linear regression model, particularly when the analytical solution is computationally challenging.

Methods Used:

Data Setup: Reused the dataset with outliers from the previous notebook, applying standard linear regression using the mean squared error (MSE) loss.
Gradient Descent Implementation: Implemented batch gradient descent with a specific learning rate, tracking the algorithm's progression over iterations to compare against the analytical solution's performance.
Key Findings:

The comparison between the analytical solution's loss and the loss achieved through gradient descent over various iteration counts provided insights into the efficiency and convergence behavior of gradient descent as an optimization method.
Conclusion:
This lab highlighted the practical application of gradient descent in linear regression, showcasing its utility in situations where the analytical solution is less feasible due to computational constraints.

Lab Notebook 11: Regularization and Logistic Regression in Linear Models
Objectives:
Extend linear regression analysis by introducing regularization techniques to combat overfitting and logistic regression for binary classification problems.

Methods Used:

Data Generation and Transformation: Generated a dataset with multiple features, including correlated features through polynomial transformations.
Modeling with Regularization: Explored Ridge regression, adjusting the regularization strength (alpha) and assessing model performance using cross-validation.
Logistic Regression Application: Implemented logistic regression for binary classification, focusing on model evaluation and the importance of feature scaling.
Key Findings:

Ridge regression's effectiveness varied with the choice of alpha, emphasizing the need for careful parameter tuning.
Logistic regression demonstrated the model's capacity for binary classification tasks, highlighting the critical role of preprocessing steps like feature scaling.
Conclusion:
The exploration of regularization and logistic regression provided deeper insights into enhancing linear regression models' performance and adaptability, underlining the significance of regularization in preventing overfitting and logistic regression in handling classification problems.

Lab Notebook 12-13: Bagging, Boosting, and Photometric Redshifts
Objectives:
This notebook focuses on utilizing ensemble methods, specifically bagging and boosting techniques like Random Forests, AdaBoost, and Gradient Boosting, to estimate photometric redshifts of galaxies based on observations in six different photometric bands.

Methods Used:

Data Preparation: Reading and preparing the data for modeling, including the transformation of features and target variables suitable for regression tasks.
Model Implementation: Application of Random Forest, AdaBoost, and Gradient Boosting models to the dataset, with an aim to reproduce and improve upon the results of a referenced paper on photometric redshift reconstruction.
Key Findings:

Ensemble methods demonstrated significant potential in improving the accuracy of photometric redshift estimation, as indicated by measures such as the normalized median absolute deviation of residuals and the fraction of outliers.
Conclusion:
The lab illustrates the power of ensemble learning methods in tackling complex regression problems in astrophysics, highlighting their effectiveness in enhancing prediction accuracy and reducing outlier fractions in the context of photometric redshift estimation.

Lab Notebook 14: Flavours of Boosting and Feature Importance
Objectives:
Expand on the use of boosting methods for estimating photometric redshifts by exploring AdaBoost and various Gradient Boosted Trees (GBM, HistGBM, and XGBoost), along with an investigation into feature importance for interpretability.

Methods Used:

Model Optimization: Thorough optimization of AdaBoost hyperparameters and exploration of different flavors of Gradient Boosted Trees to find the most effective model configurations.
Parameter Exploration: Utilization of RandomizedSearchCV for an extensive search over specified parameter values for model tuning.
Feature Importance Analysis: Examination of the feature importances provided by ensemble methods to identify the most significant predictors in the dataset.
Key Findings:

Different boosting methods showed varied performance levels, with some configurations leading to better model accuracy and lower outlier fractions.
Feature importance analysis offered insights into the predictors that played a crucial role in the models' decision-making processes, aiding in the interpretability of machine learning models in scientific research.
Conclusion:
Lab Notebook 14 demonstrates the effectiveness of advanced boosting techniques in photometric redshift estimation, emphasizing the importance of hyperparameter optimization and the value of understanding model decisions through feature importance. This approach not only enhances predictive accuracy but also contributes to the interpretability of machine learning applications in astrophysics

Lab Notebook 15: Clustering Methods
Objectives:
The aim is to understand and implement k-means++ clustering from scratch, testing the algorithm on datasets with spherically shaped and irregularly shaped clusters. Additionally, the notebook explores the use of Density-Based Spatial Clustering of Applications with Noise (DBSCAN) on these datasets to compare the effectiveness of different clustering methods.

Methods Used:

k-means++ Clustering Implementation: Writing a k-means++ clustering algorithm from scratch, focusing on optimizing initial cluster center selection to improve clustering performance.
DBSCAN Application: Applying DBSCAN to the same datasets to assess its performance, particularly in handling irregularly shaped clusters.
Key Findings:

The k-means++ algorithm demonstrated improved initialization and clustering outcomes compared to the standard k-means, particularly on spherically shaped clusters.
DBSCAN showed superior performance on datasets with irregularly shaped clusters, highlighting its advantage in identifying clusters of arbitrary shapes without specifying the number of clusters beforehand.
Conclusion:
This notebook underscores the importance of choosing the right clustering method based on dataset characteristics. k-means++ offers enhancements over standard k-means for spherical clusters, while DBSCAN provides a robust solution for datasets with irregularly shaped clusters, emphasizing the need for method selection in clustering tasks.

Lab Notebook 16: Principal Component Analysis
Objectives:
Explore Principal Component Analysis (PCA) as a tool for dimensionality reduction, applying PCA to Sloan Digital Sky Survey (SDSS) spectra and galaxy images to extract meaningful patterns and reduce data complexity.

Methods Used:

PCA Implementation: Application of PCA to high-dimensional astronomical data to identify principal components that capture the most variance.
Dimensionality Reduction: Reduction of data dimensionality to facilitate visualization, analysis, and interpretation of complex datasets.
Key Findings:

PCA effectively reduced the dimensionality of SDSS data, revealing principal components that encapsulate significant patterns and features within the data.
The reduction in dimensionality aided in the visualization and understanding of the data, highlighting the utility of PCA in managing high-dimensional datasets.
Conclusion:
Lab Notebook 16 demonstrates PCA's value in simplifying and extracting meaningful information from complex datasets, such as astronomical observations. It highlights PCA's role in enabling easier data analysis and interpretation by reducing dimensionality and focusing on the most informative features.

Lab Notebook 17: Introduction to Neural Networks
Objectives:
Introduce the basic concepts of neural networks, starting with the similarity between logistic regression and a neural network with no hidden layers, progressing to the implementation and evaluation of a simple neural network model.

Methods Used:

Logistic Regression Comparison: Implementation of logistic regression as a baseline for comparison with a neural network model.
Simple Neural Network Implementation: Use of sklearn.neural_network.MLPClassifier to create a neural network with no hidden layers, comparing its performance with logistic regression.
Key Findings:

The comparison between logistic regression and a single-layer neural network illustrated the conceptual similarity between the two, with the neural network model effectively mirroring the logistic regression model in terms of decision boundaries and predictive capabilities.
Conclusion:
This notebook serves as a foundational introduction to neural networks, establishing the link between logistic regression and neural networks. It demonstrates that a neural network without hidden layers functions similarly to logistic regression, providing a basis for understanding more complex neural network structures.









## References
- Acquaviva, V. (2023). *Machine Learning for Physics and Astronomy*. Princeton University Press. This book has been a key resource, providing important knowledge that greatly influenced the methods and techniques discussed in this repository.
