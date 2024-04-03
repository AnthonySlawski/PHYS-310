# PHYS-310
## Machine Learning for Physics and Astronomy Data Analysis 

This repository is dedicated to a collection of machine learning tasks and methodologies I learnt from my course PHYS310, covering both fundamental concepts and advanced techniques. It serves as a valuable resource for anyone interested in machine learning. Below is an outline of the key areas covered:

## Classification Tasks
- **Decision Trees:** Investigate the application of decision trees in classification assignments, a foundational method in machine learning that models decisions and their possible consequences.
- **K-Nearest-Neighbours (KNN):** Classify instances based on the closest training examples in the feature space.
- **Support Vector Machines (SVM):** Effectively perform classification by finding the optimal separating hyperplane.
- **Logistic Regression:** A fundamental statistical model used for binary classification tasks.
## Regression Tasks
- **Linear Models:** Predicting a quantitative response.
- **Regularization:** Using techniques like Lasso and Ridge regression to prevent overfitting and enhance model generalization.
- **Gradient Descent:** Crucial for learning the parameters of various models.
## Evaluation and Optimization of ML Models
- **Cross Validation:** Assessing the predictive performance of your models and ensuring they generalize well to unseen data.
- **Hyperparameter Optimization:** Optimizing the hyperparameters of your machine learning models to improve performance.
- **Feature Engineering:** Understanding the importance of feature selection and transformation in developing robust models.
## Ensemble Methods
- **Random Forests:** An ensemble learning method for classification and regression that improves predictive accuracy.
- **Boosting Methods:** Combine multiple weak learners to form a strong learner, enhancing model predictions.
## Dimensionality Reduction
- **Clustering:** Grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups.
- **Principal Component Analysis (PCA):** A technique used to emphasize variation and bring out strong patterns in a dataset.
## Neural Networks and Deep Learning
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

## [Lab Notebook 4 - Decision Trees](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%204.ipynb)

**Objective:** This notebook delves into Decision Trees (DT), a fundamental machine learning algorithm for classification and regression tasks. The exercise involves coding parts of a Decision Tree algorithm from scratch and comparing the results with a pre-written DT routine from a machine learning library.

**Methods Used:**

- Utilizes Python libraries such as NumPy for numerical operations, Pandas for data manipulation, and Matplotlib for visualization. For machine learning, it employs scikit-learn's DecisionTreeClassifier, among other utilities.
- The data preparation process mirrors that of the previous lab notebook, emphasizing the importance of loading, cleaning, and understanding the dataset before model implementation.
- The notebook guides through the implementation of a basic Decision Tree algorithm, highlighting key steps such as feature selection, tree construction, and node splitting.

**Key Findings:**

- The same dataset from the previous lab notebook is reused, ensuring continuity and providing a basis for comparing different machine learning algorithms' performance on a consistent dataset.
- Initial steps include setting up the environment, importing the necessary libraries, and preparing the dataset for analysis.

**Conclusion:** Lab Notebook 4 offers a comprehensive introduction to Decision Trees, showcasing the algorithm's logic and functionality through a hands-on coding exercise. By comparing a custom implementation with a pre-existing routine, the notebook provides insights into the algorithm's inner workings and the ease with which complex machine learning tasks can be accomplished using established libraries.

## [Lab Notebooks 5 & 6 - Advanced Machine Learning Techniques](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%205-6.ipynb)
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

## [Lab Notebook 7: Support Vector Machine (SVM)](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%207.ipynb)
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

## [Lab Notebook 8: SVM Classification in Particle Physics](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%208.ipynb)
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

## [Lab Notebook 9: Basic Linear Regression](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%209.ipynb)
**Objectives:**
Explore the basics of linear regression, including setting up a model, exploring different loss functions, and understanding the concept of residuals in relation to model fit.

**Methods Used:**

- Linear Model Setup: Utilized sklearn's LinearRegression() to fit a linear model to generated data, comparing the model's predictions against a predefined true regression line.
- Residual Analysis: Investigated the residuals - the differences between the actual data points and the model's predictions - to assess the independence of errors from the independent variable (x).

**Key Findings:**

- Residual analysis revealed that the residuals were not independent of x, indicating that the assumptions of the probabilistic linear model were not fully satisfied. However, it was still possible to create a predictive model despite this limitation.

**Conclusion:**
The lab demonstrates that while linear regression can provide a useful model for data prediction, understanding the nature of residuals and their relation to the independent variables is crucial for assessing model appropriateness and limitations.

## [Lab Notebook 10: Linear Regression with Gradient Descent](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2010.ipynb)
**Objectives:**
Introduce gradient descent as a method for finding the parameters of a linear regression model, particularly when the analytical solution is computationally challenging.

**Methods Used:**

- Data Setup: Reused the dataset with outliers from the previous notebook, applying standard linear regression using the mean squared error (MSE) loss.
- Gradient Descent Implementation: Implemented batch gradient descent with a specific learning rate, tracking the algorithm's progression over iterations to compare against the analytical solution's performance.

**Key Findings:**

- The comparison between the analytical solution's loss and the loss achieved through gradient descent over various iteration counts provided insights into the efficiency and convergence behavior of gradient descent as an optimization method.

**Conclusion:**
This lab highlighted the practical application of gradient descent in linear regression, showcasing its utility in situations where the analytical solution is less feasible due to computational constraints.

## [Lab Notebook 11: Regularization and Logistic Regression in Linear Models](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2011.ipynb)
**Objectives:**
Extend linear regression analysis by introducing regularization techniques to combat overfitting and logistic regression for binary classification problems.

**Methods Used:**

- Data Generation and Transformation: Generated a dataset with multiple features, including correlated features through polynomial transformations.
- Modeling with Regularization: Explored Ridge regression, adjusting the regularization strength (alpha) and assessing model performance using cross-validation.
- Logistic Regression Application: Implemented logistic regression for binary classification, focusing on model evaluation and the importance of feature scaling.

**Key Findings:**

- Ridge regression's effectiveness varied with the choice of alpha, emphasizing the need for careful parameter tuning.
- Logistic regression demonstrated the model's capacity for binary classification tasks, highlighting the critical role of preprocessing steps like feature scaling.

**Conclusion:**
The exploration of regularization and logistic regression provided deeper insights into enhancing linear regression models' performance and adaptability, underlining the significance of regularization in preventing overfitting and logistic regression in handling classification problems.

## [Lab Notebook 12-13: Bagging, Boosting, and Photometric Redshifts](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2012-13.ipynb)
**Objectives:**
This notebook focuses on utilizing ensemble methods, specifically bagging and boosting techniques like Random Forests, AdaBoost, and Gradient Boosting, to estimate photometric redshifts of galaxies based on observations in six different photometric bands.

**Methods Used:**

- Data Preparation: Reading and preparing the data for modeling, including the transformation of features and target variables suitable for regression tasks.
- Model Implementation: Application of Random Forest, AdaBoost, and Gradient Boosting models to the dataset, with an aim to reproduce and improve upon the results of a referenced paper on photometric redshift reconstruction.

**Key Findings:**

- Ensemble methods demonstrated significant potential in improving the accuracy of photometric redshift estimation, as indicated by measures such as the normalized median absolute deviation of residuals and the fraction of outliers.

**Conclusion:**
The lab illustrates the power of ensemble learning methods in tackling complex regression problems in astrophysics, highlighting their effectiveness in enhancing prediction accuracy and reducing outlier fractions in the context of photometric redshift estimation.

## [Lab Notebook 14: Flavours of Boosting and Feature Importance](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2014.ipynb)
**Objectives:**
- Expand on the use of boosting methods for estimating photometric redshifts by exploring AdaBoost and various Gradient Boosted Trees (GBM, HistGBM, and XGBoost), along with an investigation into feature importance for interpretability.

**Methods Used:**

- Model Optimization: Thorough optimization of AdaBoost hyperparameters and exploration of different flavors of Gradient Boosted Trees to find the most effective model configurations.
- Parameter Exploration: Utilization of RandomizedSearchCV for an extensive search over specified parameter values for model tuning.
- Feature Importance Analysis: Examination of the feature importances provided by ensemble methods to identify the most significant predictors in the dataset.

**Key Findings:**

- Different boosting methods showed varied performance levels, with some configurations leading to better model accuracy and lower outlier fractions.
- Feature importance analysis offered insights into the predictors that played a crucial role in the models' decision-making processes, aiding in the interpretability of machine learning models in scientific research.

**Conclusion:**
- Lab Notebook 14 demonstrates the effectiveness of advanced boosting techniques in photometric redshift estimation, emphasizing the importance of hyperparameter optimization and the value of understanding model decisions through feature importance. This approach not only enhances predictive accuracy but also contributes to the interpretability of machine learning applications in astrophysics

## [Lab Notebook 15: Clustering Methods](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2015.ipynb)
**Objectives:**
The aim is to understand and implement k-means++ clustering from scratch, testing the algorithm on datasets with spherically shaped and irregularly shaped clusters. Additionally, the notebook explores the use of Density-Based Spatial Clustering of Applications with Noise (DBSCAN) on these datasets to compare the effectiveness of different clustering methods.

**Methods Used:**

- k-means++ Clustering Implementation: Writing a k-means++ clustering algorithm from scratch, focusing on optimizing initial cluster center selection to improve clustering performance.
- DBSCAN Application: Applying DBSCAN to the same datasets to assess its performance, particularly in handling irregularly shaped clusters.

**Key Findings:**

- The k-means++ algorithm demonstrated improved initialization and clustering outcomes compared to the standard k-means, particularly on spherically shaped clusters.
- DBSCAN showed superior performance on datasets with irregularly shaped clusters, highlighting its advantage in identifying clusters of arbitrary shapes without specifying the number of clusters beforehand.

**Conclusion:**
This notebook underscores the importance of choosing the right clustering method based on dataset characteristics. k-means++ offers enhancements over standard k-means for spherical clusters, while DBSCAN provides a robust solution for datasets with irregularly shaped clusters, emphasizing the need for method selection in clustering tasks.

## [Lab Notebook 16: Principal Component Analysis](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2016.ipynb)
**Objectives:**
Explore Principal Component Analysis (PCA) as a tool for dimensionality reduction, applying PCA to Sloan Digital Sky Survey (SDSS) spectra and galaxy images to extract meaningful patterns and reduce data complexity.

**Methods Used:**

- PCA Implementation: Application of PCA to high-dimensional astronomical data to identify principal components that capture the most variance.
- Dimensionality Reduction: Reduction of data dimensionality to facilitate visualization, analysis, and interpretation of complex datasets.

**Key Findings:**

- PCA effectively reduced the dimensionality of SDSS data, revealing principal components that encapsulate significant patterns and features within the data.
- The reduction in dimensionality aided in the visualization and understanding of the data, highlighting the utility of PCA in managing high-dimensional datasets.

**Conclusion:**
Lab Notebook 16 demonstrates PCA's value in simplifying and extracting meaningful information from complex datasets, such as astronomical observations. It highlights PCA's role in enabling easier data analysis and interpretation by reducing dimensionality and focusing on the most informative features.

## [Lab Notebook 17: Introduction to Neural Networks](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2017.ipynb)
**Objectives:**
Introduce the basic concepts of neural networks, starting with the similarity between logistic regression and a neural network with no hidden layers, progressing to the implementation and evaluation of a simple neural network model.

**Methods Used:**

- Logistic Regression Comparison: Implementation of logistic regression as a baseline for comparison with a neural network model.
- Simple Neural Network Implementation: Use of sklearn.neural_network.MLPClassifier to create a neural network with no hidden layers, comparing its performance with logistic regression.

**Key Findings:**

- The comparison between logistic regression and a single-layer neural network illustrated the conceptual similarity between the two, with the neural network model effectively mirroring the logistic regression model in terms of decision boundaries and predictive capabilities.

**Conclusion:**
This notebook serves as a foundational introduction to neural networks, establishing the link between logistic regression and neural networks. It demonstrates that a neural network without hidden layers functions similarly to logistic regression, providing a basis for understanding more complex neural network structures.

## [Lab Notebook 18: Introduction to Neural Networks](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2018.ipynb)
**Objectives:**
The lab introduces neural networks by illustrating the relationship between logistic regression and a single-layer neural network (i.e., a neural network with no hidden layers).

**Methods Used:**

- Comparison with Logistic Regression: Employed logistic regression on a dataset to serve as a baseline for comparison.
- Single Neuron Model: Implemented a neural network model using MLPClassifier from sklearn.neural_network with no hidden layers and compared its performance to logistic regression.

**Key Findings:**

- The single neuron model, essentially a neural network with no hidden layers, demonstrated a performance closely mirroring that of logistic regression, indicating their conceptual similarity.

**Conclusion:**
This lab effectively introduces neural networks by drawing a parallel with logistic regression, showing that at their simplest form, neural networks can replicate logistic regression's functionality, providing a foundational understanding for more complex neural network architectures.

## [Lab Notebook 19: Particle ID Classification with Neural Networks](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2019.ipynb)
**Objectives:**
Develop a fully connected neural network to address the particle identification (ID) classification problem, comparing its performance with previously obtained results using support vector machines (SVM).

**Methods Used:**

- Dataset Preparation: Loaded and prepared the dataset for neural network training, including data shuffling and splitting into training, validation, and test sets.
- Neural Network Implementation: Built and trained a fully connected neural network on the particle ID classification task, focusing on optimizing model parameters and architecture.

**Key Findings:**

- The fully connected neural network aimed to surpass the performance of an optimal SVM model, which previously achieved an accuracy of 94-95% on the same problem.

**Conclusion:**
This notebook highlights the application of neural networks in solving classification problems in physics, demonstrating their potential to achieve high accuracy in complex tasks such as particle ID classification.

## [Lab Notebook 20: Photometric Redshift Estimation with Deep Neural Networks](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2020.ipynb)
**Objectives:**
Utilize a fully connected deep neural network to tackle the photometric redshift estimation problem, exploring the impact of different loss functions and learning rate schedules on model performance.

**Methods Used:**

- Dataset Utilization: Engaged with a high-quality dataset previously employed in ensemble methods, aiming to achieve or improve upon the best model's outlier fraction of 4%.
- Deep Neural Network Deployment: Implemented a deep neural network for regression, adjusting the loss function and learning rate schedule to optimize performance.

**Key Findings:**

-The deep neural network's objective was to match or exceed the performance of the best ensemble method model, focusing on reducing the outlier fraction in photometric redshift estimation.

**Conclusion:**
Lab Notebook 20 explores the effectiveness of deep neural networks in addressing regression problems within astrophysics, emphasizing the importance of model tuning and the selection of appropriate loss functions and learning rate schedules for optimal performance. ​

## [Lab Notebook 21-22: Discovery of Exoplanets](https://github.com/AnthonySlawski/PHYS-310/blob/main/Lab%20Notebook%2021-22.ipynb)
**Objectives:**
The lab focuses on detecting exoplanets using data from the NASA Kepler space telescope, employing the transit method which involves analyzing the brightness of stars over time to identify periodic dimming caused by planet transits.

**Methods Used:**

- Data Preprocessing: Reading the dataset into pandas, performing summary statistics, and reformatting data for model input.
- Convolutional Neural Network (CNN) Training: Preparing the data for CNNs through Fast Fourier Transform (FFT), scaling, filtering, and subsequently training a CNN to classify exoplanet and non-exoplanet stars based on flux intensity patterns.

**Key Findings:**

- The lab illustrated the challenges associated with exoplanet detection, notably due to the rarity of exoplanets and the resulting dataset imbalance.
- CNNs, with proper preprocessing steps like FFT and data scaling, demonstrated potential in identifying exoplanets by learning patterns in flux intensity over time.

**Conclusion:**
This notebook showcases the application of CNNs in astrophysics for the challenging task of exoplanet discovery. Through careful data preprocessing and model training, CNNs can effectively identify exoplanets, underscoring the importance of advanced machine learning techniques in astronomical observations.

## [Homework 1: Exploring Classifiers](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%201.ipynb)
**Objectives:**
Homework 1 investigates the performance of k-nearest neighbors (kNN) and decision tree classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset from sklearn, emphasizing model accuracy and feature importance.

**Methods Used:**

- Model Evaluation: Application of kNN and decision tree classifiers to the dataset, with accuracy assessment across various parameter settings.
- Feature Importance Analysis: Utilization of decision tree's feature_importances_ to understand the most significant features in the classification process.

**Key Findings:**

- The optimal settings for both kNN and decision tree classifiers were identified through accuracy score analysis, revealing the impact of parameters like k in kNN and max_depth, max_features in decision trees on model performance.
- Feature importance analysis provided insights into the most critical features for cancer diagnosis, offering potential guidance for medical practitioners.

**Conclusion:**
Homework 1 illustrates the utility of kNN and decision tree classifiers in medical diagnosis, highlighting the importance of parameter tuning and feature importance analysis. It demonstrates how machine learning can assist in identifying the most relevant features for accurate disease classification.

## [Homework 2: Experiments with kNN and Decision Tree Classifiers](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%202.ipynb)
**Objectives:**
To explore and evaluate the performance of k-nearest neighbors (kNN) and decision tree classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset from sklearn, with a focus on optimizing model parameters for improved accuracy.

**Methods Used:**

- Classifier Application: Applied kNN and decision tree classifiers to the dataset, including parameter optimization for kNN (number of neighbors) and decision trees (max depth and max features).
- Accuracy Evaluation: Evaluated model accuracy across different parameter settings, employing techniques like scaling with StandardScaler to assess its impact on kNN performance.
- Feature Importance Analysis: Leveraged decision tree's feature_importances_ to identify and visualize the most significant features in predicting cancer malignancy.

**Key Findings:**

- The optimal k value for kNN and the best combination of max depth and max features for decision trees were identified, significantly impacting model accuracy.
- Feature scaling improved kNN performance, highlighting the importance of preprocessing in machine learning workflows.
- Feature importance analysis from the decision tree model provided insights into critical predictors of cancer malignancy, offering potential clinical relevance.

**Conclusion:**
This homework demonstrates the effectiveness of kNN and decision tree classifiers in medical diagnosis problems, emphasizing the importance of parameter tuning and feature scaling. The analysis of feature importance further underscores the potential of decision trees in providing interpretable machine learning models.

## [Homework 3: Classification and Prediction in Astronomy](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%203.ipynb)
**Objectives:**
The assignment focuses on classifying and predicting whether a star is a RR-Lyrae variable star based on its color features, addressing key questions to understand the dataset and the nature of the prediction task.

**Methods Used:**

- Data Exploration: Loaded and examined the dataset to understand its structure, including the number of instances, features, and the proportion of RR-Lyrae stars.
- Problem Identification: Determined the nature of the problem as a classification task, identifying it as supervised learning.

**Key Findings:**

- The task was recognized as a binary classification problem in a supervised learning context, aiming to predict the presence of RR-Lyrae variable stars from color features.
- An understanding of the dataset's balance and the baseline accuracy achievable by a naive classifier was established, setting the stage for more sophisticated analysis.

**Conclusion:**
Homework 3 lays the groundwork for applying machine learning models to astronomical data, highlighting the importance of initial data exploration and problem framing in guiding subsequent analysis and model selection.

## [Homework 4: Discover the Higgs Boson!](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%204.ipynb)
**Objectives:**
Engage with a simplified version of the simulated Higgs boson data challenge to classify instances as either "no Higgs signal" or "Higgs signal," using provided features.

**Methods Used:**

- Data Loading and Exploration: Loaded the feature and label datasets, examining the number of instances and features available for analysis.
- Feature Distribution Analysis: Analyzed and visualized the distribution of each feature within the dataset to understand the data's structure and variability.

**Key Findings:**

- Initial exploration provided a clear overview of the dataset's composition, including the number of features and instances, essential for planning subsequent analysis.
- Visualization of feature distributions offered insights into the characteristics of the data, potentially informing feature selection and preprocessing decisions in model development.

**Conclusion:**
Homework 4 introduces participants to the complexities and challenges of analyzing high-energy physics data, emphasizing the importance of thorough data exploration and visualization in understanding the underlying patterns that may signal the presence of the Higgs boson.

## [Homework 5: Learning the Ising Model Coupling Constants](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%205.ipynb)
**Objectives:**
To apply linear regression techniques to learn the coupling strengths $J_{j,k}$ of the one-dimensional Ising model, a fundamental model in statistical mechanics describing interactions between spins in a lattice.

**Methods Used:**

- Model Formulation: Recast the Ising model in a form suitable for linear regression, representing spin interactions as vectors and the task as learning the coupling constants.
- Linear Regression Application: Used linear regression to estimate the coupling constants based on given spin configurations and their computed energies.

**Key Findings:**

- Demonstrated that linear regression could effectively learn the non-local coupling strengths of the Ising model from data, revealing the model's ability to capture the underlying physics of spin interactions.

**Conclusion:**
This homework showcases the intersection of physics and machine learning, highlighting how linear regression can uncover the parameters governing physical systems. It underlines the model's potential to learn complex interactions from data, offering insights into the mechanics of the Ising model.

## [Homework 6: Variance of Correlated Variables and Optimization of Extremely Random Trees](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%206%20.ipynb)
**Objectives:**
- Variance Proof: Prove that the variance of a sum of correlated variables can be expressed in a specific form, highlighting the impact of correlation on variance.
- Algorithm Optimization: Optimize the Extremely Random Tree algorithm for predicting photometric redshifts, comparing its performance to the optimal Random Forest model.
  
**Methods Used:**

- Utilized properties of covariance to derive the variance formula, considering both the correlation coefficient and the variance of individual variables.
- Employed grid search for hyperparameter optimization of the Extremely Random Tree model, assessing performance through metrics such as mean absolute error and outlier fraction.

**Key Findings:**
  
- The derived variance formula underscores the influence of correlation among variables on the overall variance of their sum.
- The Extremely Random Tree algorithm demonstrated superior performance compared to the Random Forest model in terms of lower error rates and outlier fractions, suggesting it as a more effective model for this task.

**Conclusion:**
Homework 6 illustrates the significant impact of variable correlation on statistical variance and showcases the advantages of using Extremely Random Trees for complex prediction tasks. Through careful mathematical derivation and model optimization, the homework not only provides insights into statistical principles but also enhances machine learning model performance, with the Extremely Random Tree model emerging as the preferred choice for predicting photometric redshifts due to its lower error rates and computational efficiency.

## [Homework 7: Hyperparameter Optimization with Keras Tuner](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%207.ipynb)
**Objectives:**
Explore hyperparameter optimization for neural networks using Keras Tuner, focusing on improving model performance for a regression problem—specifically, the estimation of photometric redshifts.

**Methods Used:**

- Keras Tuner Implementation: Utilized Keras Tuner for systematic exploration of hyperparameters, including the number and size of layers and the learning rate, to find the optimal network configuration.
- Model Evaluation: Assessed the performance of the optimized model on a test set, using metrics relevant to the task (outlier fraction and normalized median absolute deviation).

**Key Findings:**

- The tuning process identified several optimal configurations, underscoring the importance of model complexity and learning rate in achieving high prediction accuracy.
- The best-performing model, as determined by Keras Tuner, showed improved performance over baseline models, demonstrating the effectiveness of hyperparameter tuning in enhancing model outcomes.

**Conclusion:**
Homework 7 illustrates the power of hyperparameter tuning in refining neural network models for complex tasks. Keras Tuner emerged as a valuable tool for navigating the vast hyperparameter space, leading to significant improvements in model accuracy and efficiency in predicting photometric redshifts.

## [Homework 8: Hyperparameter Optimization with Keras Tuner and Cross-Validation with SciKeras](https://github.com/AnthonySlawski/PHYS-310/blob/main/Homework%208.ipynb)
**Objectives:**
The goal of this homework is to explore hyperparameter optimization strategies for neural networks in the context of the photometric redshift estimation problem. It also introduces the use of cross-validation (CV) with the SciKeras wrapper to enhance model validation processes.

**Methods Used:**

- Keras Tuner for Hyperparameter Optimization: Employed Keras Tuner to systematically explore and optimize neural network configurations, including variations in the number of layers, the number of neurons per layer, and learning rates.
- Model Definition and Tuning: Defined a neural network model function that is compatible with Keras Tuner, specifying the architecture and hyperparameters to be tuned.
- Evaluation and Validation: Utilized the best model configurations identified by Keras Tuner to fit the data, employing cross-validation techniques with SciKeras to rigorously evaluate model performance.

**Key Findings:**

- Keras Tuner facilitated the identification of optimal hyperparameters, potentially improving upon previous models' performance by exploring a wide range of network configurations.
- The application of cross-validation with SciKeras provided a more robust evaluation of the neural network model, ensuring that the performance assessment was thorough and less prone to overfitting.

**Conclusion:**
Homework 8 demonstrates the effectiveness of advanced tools like Keras Tuner and SciKeras in refining neural network models for complex regression tasks. Through strategic hyperparameter tuning and enhanced validation methods, it is possible to significantly improve model performance, underscoring the value of these techniques in machine learning workflows.

## References
- Acquaviva, V. (2023). *Machine Learning for Physics and Astronomy*. Princeton University Press. This book has been a key resource, providing important knowledge that greatly influenced the methods and techniques discussed in this repository.
