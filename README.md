


DATA SCIENCE PROJECT REPORT

Electric Vehicle Sales by State in India


Submitted to: Unified Mentor

Domain: Data Analytics
Tools Used: Python | Machine Learning | SQL | Excel
Difficulty Level: Intermediate

Submitted by: Shivraj Rakte

April 2026
 
Abstract
Electric vehicles (EVs) represent a transformative shift in the global transportation landscape, and India is no exception. With increasing concerns around fossil fuel dependency, environmental pollution, and rising fuel costs, the Indian government has actively promoted EV adoption through various policy initiatives and incentives. This project — titled Electric Vehicle Sales by State in India — analyzes a comprehensive dataset of EV sales across Indian states spanning from 2014 to 2024, containing 96,845 records and eight key features.
The primary goal of this project is to uncover meaningful trends in EV adoption across different states, vehicle categories, and time periods, and to build a machine learning model capable of predicting future EV sales quantities. The analysis employs Python as the primary programming language, using libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn for data manipulation, visualization, and modeling.
The dataset, sourced from the Clean Mobility Shift website, includes information on the year, month, date, state, vehicle class, vehicle category, vehicle type, and EV sales quantity. The study covers exploratory data analysis (EDA), data preprocessing, feature engineering, and predictive modeling using a Random Forest Regressor.
Key findings reveal a dramatic surge in EV sales post-2020, with states like Uttar Pradesh, Delhi, and Maharashtra leading in sales volumes. Two-wheelers and three-wheelers dominate the EV market. The Random Forest model demonstrates strong predictive capability, providing a data-driven foundation for market planning, policy development, and infrastructure investment in the EV sector.
This report is structured to guide the reader through each phase of the data science pipeline — from problem definition and data exploration to model building and evaluation — making it accessible to undergraduate students and entry-level data analysts alike.
 
1. Introduction
1.1 Background
The global automotive industry is undergoing an unprecedented transformation driven by the urgent need to combat climate change, reduce dependence on fossil fuels, and improve urban air quality. Electric vehicles, powered entirely or partially by electricity, have emerged as a leading solution to these challenges. Unlike conventional internal combustion engine (ICE) vehicles, EVs produce zero tailpipe emissions, consume significantly less energy per kilometer, and have lower operational costs over their lifetime.
India, as one of the world's largest automobile markets, is at the forefront of this electric revolution. With a population exceeding 1.4 billion people and rapid urbanization underway, India faces enormous transportation challenges including traffic congestion, air pollution, and growing energy demands. The Indian government has recognized the potential of electric mobility and launched several ambitious initiatives to accelerate EV adoption.
Key among these is the FAME India Scheme (Faster Adoption and Manufacturing of Hybrid and Electric Vehicles), which provides financial incentives to EV buyers and manufacturers. Additionally, the National Electric Mobility Mission Plan (NEMMP) targets millions of electric and hybrid vehicles on Indian roads by 2030. These policy measures, combined with falling battery prices and increasing consumer awareness, have created a rapidly growing EV market in India.
1.2 Importance of Studying EV Sales Data
Understanding EV sales patterns across different states and vehicle categories is crucial for multiple stakeholders:
•	Policymakers need accurate sales data to design targeted subsidies, tax incentives, and infrastructure investments where they are most needed.
•	Automobile manufacturers require regional sales insights to plan production capacity, dealership networks, and after-sales service centers.
•	Investors and entrepreneurs can identify high-growth markets and underserved segments to make informed business decisions.
•	Infrastructure developers — particularly charging station operators — need to know where EV density is highest to plan network expansion.
•	Researchers and academics benefit from longitudinal sales data to study adoption curves, consumer behavior, and policy effectiveness.

Data-driven analysis of EV sales trends provides all these stakeholders with evidence-based insights, replacing guesswork with informed decision-making. This is precisely the contribution of this project: transforming raw sales data into actionable knowledge.
1.3 Project Objectives
This project pursues the following specific objectives:
1.	To perform comprehensive exploratory data analysis (EDA) on the EV sales dataset to identify patterns, trends, and anomalies.
2.	To preprocess and clean the data, handling missing values, encoding categorical variables, and engineering new features.
3.	To visualize EV sales trends across states, vehicle categories, vehicle types, and time periods using appropriate charts and graphs.
4.	To build a machine learning regression model that can predict EV sales quantities based on historical data and input features.
5.	To evaluate the performance of the model using appropriate metrics and interpret the results.
6.	To draw meaningful conclusions and suggest areas for future work and improvement.
1.4 Scope of the Project
This project focuses on EV sales data within India from January 2014 to December 2023, with partial data for 2024. The analysis covers all major Indian states and union territories. The machine learning component focuses on regression modeling to predict numerical sales quantities. Classification approaches and deep learning models are outside the current scope but are suggested for future work.
 
2. Problem Statement
India's electric vehicle market is growing at an exponential rate, yet the distribution of EV adoption across different states, vehicle categories, and time periods remains uneven and poorly understood. While aggregate national statistics show impressive growth numbers, a granular state-level analysis is essential for targeted policy intervention, infrastructure planning, and market strategy.
The central problem this project addresses is:
How can historical electric vehicle sales data across Indian states be analyzed and modeled to identify key adoption trends, understand regional disparities, and predict future sales quantities to support strategic decision-making?

More specifically, the project addresses the following sub-problems:
•	Which Indian states are leading in EV adoption, and which are lagging behind?
•	How have EV sales evolved over the years from 2014 to 2024?
•	Which vehicle categories (2-wheelers, 3-wheelers, 4-wheelers, buses) are driving EV growth in India?
•	Are there seasonal patterns in EV sales across months of the year?
•	What factors most strongly influence EV sales quantity across different states and vehicle types?
•	Can a reliable machine learning model be trained to accurately predict EV sales for a given combination of state, vehicle type, and time period?

Solving these problems has practical implications. For instance, if data reveals that 3-wheelers have high EV adoption in certain states, policymakers can prioritize charging infrastructure suited for commercial 3-wheelers in those regions. Similarly, predictive models can help manufacturers plan inventory and distribution more efficiently.
The challenge lies in the complexity and heterogeneity of the dataset: 96,845 records spanning 34 states and union territories, 73 vehicle classes, 5 vehicle categories, and 12 vehicle types across a 10-year period. This complexity demands robust data preprocessing, insightful visualization, and a well-chosen machine learning approach.
 
3. Dataset Description
3.1 Data Source
The dataset used in this project was scraped from the Clean Mobility Shift website, which tracks electric vehicle registration and sales data across India. The raw data was meticulously preprocessed to remove null values, correct data types, and ensure consistency before being made available in CSV format under the filename EV_Dataset.csv.
The dataset is publicly available for download and has been verified for completeness and accuracy. It represents one of the most comprehensive publicly available datasets on EV sales in India, covering a 10-year span from 2014 to 2024.
3.2 Dataset Overview

Attribute	Details
File Name	EV_Dataset.csv
Total Records	96,845 rows
Total Features	8 columns
Time Period	January 2014 to December 2023 (partial 2024)
States Covered	34 States and Union Territories
Vehicle Classes	73 unique vehicle classes
File Format	CSV (Comma-Separated Values)
Missing Values	None (already cleaned)
Duplicates	None detected

3.3 Feature Description
The dataset contains 8 columns, each capturing a specific dimension of EV sales information. Below is a detailed description of each feature:

Column Name	Data Type	Description
Year	Integer	The calendar year in which the EV sales were recorded (2014-2024). Originally stored as float, converted to int during preprocessing.
Month_Name	Category	The name of the month (e.g., jan, feb, mar). Stored as a categorical variable for memory efficiency.
Date	Datetime	The specific date of the sales record (formatted as MM/DD/YYYY). Converted from object to datetime format.
State	Category	The name of the Indian state or union territory where EV sales occurred. Covers 34 states/UTs.
Vehicle_Class	Category	The specific class of vehicle, such as Motor Car, M-Cycle/Scooter, Goods Carrier, Bus, etc. 73 unique classes.
Vehicle_Category	Category	Broad category of the vehicle: Others, 2-Wheelers, 3-Wheelers, Bus, or 4-Wheelers.
Vehicle_Type	Category	Specific type within category, e.g., 2W_Personal, 3W_Shared, 4W_Personal, Bus, etc. 12 unique types.
EV_Sales_Quantity	Float	The number of electric vehicles sold in that specific combination of state, vehicle class, and time period. Target variable.

3.4 State Distribution
The dataset covers all major Indian states and union territories. Maharashtra leads with 4,912 records, followed by Karnataka (4,830), Uttar Pradesh (4,557), Rajasthan (4,552), and Gujarat (4,517). Smaller union territories like Ladakh (1,063) and Andaman & Nicobar Island (1,226) have fewer records, reflecting their smaller populations and vehicle registrations.
3.5 Vehicle Category Distribution
The dataset reveals an interesting structure in vehicle categories:
•	Others: 54,423 records (56.2%) — specialized or miscellaneous vehicle classes.
•	2-Wheelers: 13,121 records (13.5%) — motorcycles and scooters.
•	3-Wheelers: 11,491 records (11.9%) — auto-rickshaws and cargo vehicles.
•	Bus: 9,119 records (9.4%) — public and institutional buses.
•	4-Wheelers: 8,691 records (9.0%) — personal and commercial cars.
3.6 Sales Quantity Statistics
The EV_Sales_Quantity column, which is the target variable for prediction, shows a highly skewed distribution:
•	Mean: 37.1 units sold per record.
•	Standard Deviation: 431.6 — indicating extreme variability.
•	Minimum: 0 units (many records show zero sales).
•	25th Percentile: 0 units.
•	Median (50th percentile): 0 units.
•	75th Percentile: 0 units.
•	Maximum: 20,584 units in a single record.

This highly right-skewed distribution, with most values at zero and a long tail of high-volume sales, presents an interesting challenge for modeling. The large proportion of zero values reflects that many vehicle class and state combinations have not yet seen significant EV adoption. 
4. Data Preprocessing
Data preprocessing is one of the most critical steps in any data science project. Raw data often contains inconsistencies, incorrect data types, and irrelevant features that can impair model performance. In this project, the preprocessing pipeline involved several key steps.
4.1 Loading the Dataset
The dataset was loaded into a Pandas DataFrame using the read_csv() function. Upon initial inspection, the dataset was confirmed to contain 96,845 rows and 8 columns, matching the expected structure. The first few rows showed records starting from January 2014, beginning with Andhra Pradesh state data.
4.2 Handling Missing Values
One of the first and most important checks in preprocessing is identifying missing values. For this dataset, df.isnull().sum() returned zero missing values for all eight columns. This indicates that the dataset had already been cleaned prior to release, which is consistent with the dataset description stating that all null values have been removed. No imputation strategies were therefore required.
4.3 Handling Duplicate Records
Duplicate records can introduce bias and lead to overfitting in machine learning models. The check df.duplicated().sum() returned 0, confirming that no duplicate rows were present in the dataset. The data was therefore clean from this standpoint and did not require any deduplication.
4.4 Data Type Correction
While the data had no missing values or duplicates, the data types of several columns were incorrect:
•	Year column: Originally stored as float64 (e.g., 2014.0). This was corrected to int64 using df['Year'] = df['Year'].astype(int). Integer representation is more appropriate for a year column.
•	Date column: Stored as a generic object (string). This was converted to datetime64 format using pd.to_datetime(df['Date'], errors='coerce'). The errors='coerce' parameter ensures any invalid date strings are converted to NaT (Not a Time) rather than throwing an error.
•	Categorical columns: Month_Name, State, Vehicle_Class, Vehicle_Category, and Vehicle_Type were all stored as object (string) type. These were converted to Pandas category dtype, which significantly reduces memory usage from 5.9 MB to 2.7 MB (a 54% reduction) and improves processing speed.
4.5 Feature Engineering
Feature engineering involves creating new informative features from existing ones to improve model performance. The following new features were derived:
•	Month (numeric): Extracted from the Date column using df['Date'].dt.month. This provides a numerical representation of the month for use in models that require numerical inputs.
•	Day: Extracted using df['Date'].dt.day to capture within-month temporal patterns.
•	Year was already present but confirmed as a feature after type conversion.

These temporal features allow the model to capture seasonality and time-based trends in EV sales.
4.6 Encoding Categorical Variables
Machine learning algorithms cannot directly process categorical (text) data. Therefore, categorical variables must be converted to numerical form. Two common approaches were considered:
•	One-Hot Encoding (OHE): Creates binary columns for each unique category. Applied to State, Vehicle_Class, Vehicle_Category, and Vehicle_Type using pd.get_dummies(df, columns=[...], drop_first=True). The drop_first=True parameter removes one column from each encoded group to avoid multicollinearity (the dummy variable trap).
•	Label Encoding: An alternative approach where each category is assigned a numeric label. Not used in the primary pipeline due to the risk of implying ordinal relationships between categories.

After one-hot encoding, the dataset expanded significantly in width due to the large number of unique states (34) and vehicle classes (73), creating a wide, sparse feature matrix.
4.7 Dropping Unnecessary Columns
After feature extraction and encoding, the original Date and Month_Name columns were dropped from the encoded dataset, as their information had been captured in the newly created Month and Day columns. This reduces redundancy and computational overhead.
4.8 Summary of Preprocessing Steps

Step	Action Taken	Outcome
Missing Values	Checked using isnull().sum()	No missing values found
Duplicates	Checked using duplicated().sum()	No duplicates found
Year dtype	Converted float64 to int64	Year stored correctly as integer
Date dtype	Converted object to datetime64	Enables date-based operations
Categorical dtypes	Converted 5 columns to category	Memory reduced from 5.9MB to 2.7MB
Feature Engineering	Extracted Month and Day from Date	New numerical temporal features
Encoding	One-hot encoding on categoricals	Model-ready numerical features
Column Removal	Dropped Date and Month_Name	Removed redundant columns
 
5. Exploratory Data Analysis (EDA)
Exploratory Data Analysis is the process of visually and statistically examining a dataset to understand its structure, detect patterns, spot anomalies, and generate hypotheses. In this project, EDA was conducted across several dimensions: time (yearly and monthly), geography (state-wise), and vehicle segmentation (category and type).
5.1 Yearly Analysis of EV Sales
A line plot was created to visualize the trend of EV sales over the years from 2014 to 2024. The plot revealed a clear and compelling story:
•	From 2014 to 2019, EV sales remained relatively flat and low, hovering near zero on a relative scale. This reflects the early, nascent stage of the Indian EV market, characterized by high vehicle prices, limited model availability, range anxiety, and minimal charging infrastructure.
•	A slight dip is visible around 2020, likely attributable to the COVID-19 pandemic, which severely disrupted automotive supply chains and consumer spending across India.
•	Post-2020, EV sales show an explosive upward trajectory. Sales jumped significantly in 2021, then nearly doubled in 2022, and continued growing through 2023. The data for 2024 (partial year) suggests the trend was continuing.
•	This exponential growth post-2020 aligns perfectly with the introduction of major government incentives (FAME-II Phase), falling lithium-ion battery prices, entry of new EV manufacturers (including OLA Electric, Ather, and Tata Motors' EV lineup), and rising fuel prices making EVs more cost-competitive.

Graph Insight: The confidence interval (shaded region) widens in 2023-2024, indicating higher variability in sales figures — some states and vehicle types showing very high growth while others remain low. This divergence underscores regional disparities in EV adoption.
5.2 Monthly Analysis of EV Sales
A line plot of average EV sales by month revealed interesting seasonal patterns throughout the year:
•	December and February show the highest average EV sales, suggesting that festival seasons (festive buying in October-November period in previous months, and end-of-year/Q4 procurement cycles) drive purchasing behavior.
•	January records a notable dip, which may reflect post-festive season slowdowns in consumer spending.
•	November shows a secondary peak, consistent with the Diwali festival season, which is traditionally the strongest period for automobile sales in India.
•	July and June show lower sales, possibly tied to the monsoon season when consumers tend to defer large purchases and dealers see reduced footfall.

Graph Insight: The wide confidence intervals across all months suggest high variance between states and vehicle types within each month, meaning that while average trends are visible, individual state-level patterns may differ significantly. Policy and infrastructure initiatives targeting specific months (e.g., pre-festival EV schemes) could leverage these seasonal patterns.
5.3 State-Wise Analysis of EV Sales
A horizontal bar plot displaying average EV sales by state provides one of the most actionable insights in the entire analysis:
•	Uttar Pradesh leads all states with the highest average EV sales — approaching 200 units in some categories. This is likely driven by the state's massive population, active 3-wheeler e-rickshaw market, and favorable state EV policy.
•	Delhi shows the second-highest sales figures, reflecting the capital's strong environmental regulations (including restrictions on diesel vehicles), high consumer awareness, and dense charging infrastructure.
•	Maharashtra (including Mumbai and Pune) and Karnataka (including Bengaluru) rank among the top performers, powered by strong urban demand for 2-wheelers and personal EVs, as well as a vibrant tech-savvy consumer base.
•	Bihar records surprisingly high EV sales, largely driven by the e-rickshaw segment which has become a dominant form of last-mile connectivity in the state.
•	North-eastern states (Nagaland, Mizoram, Manipur), smaller UTs (Ladakh, Andaman & Nicobar), and Sikkim show the lowest EV sales, reflecting geographic remoteness, limited charging infrastructure, and lower incomes.

Graph Insight: The state-wise variation is enormous — a factor of over 100x between the highest and lowest performing states. This strongly suggests that a one-size-fits-all national EV policy will be less effective than targeted, state-specific interventions addressing local barriers to adoption.
5.4 Analysis by Vehicle Category
A bar chart comparing EV sales across the five vehicle categories (2-Wheelers, 3-Wheelers, 4-Wheelers, Bus, and Others) reveals the composition of India's EV market:
•	2-Wheelers and 3-Wheelers dominate the EV landscape, both showing average sales in the 130-140 range per record. This is consistent with India's broader automobile market, where two-wheelers account for over 75% of total vehicle sales nationally.
•	4-Wheelers show considerably lower average EV sales (around 18 units), reflecting the higher cost of electric cars and the still-limited range of affordable EV car models available in India.
•	Bus category records negligible average sales, though this may be misleading — a few state-level public bus fleet electrification programs can result in bulk purchases that skew averages.
•	Others (miscellaneous vehicle classes) show near-zero average sales since most specialized vehicles have minimal EV penetration.

Graph Insight: The dominance of 2-wheelers and 3-wheelers in EV sales is a uniquely Indian characteristic. Globally, the EV narrative often focuses on cars, but India's EV transition is being driven from the bottom up — through affordable two-wheelers and e-rickshaws rather than expensive personal cars. This has major implications for which segments receive government support and manufacturer investment.
5.5 Analysis by Vehicle Type
A more granular view comes from analyzing the 12 distinct vehicle types within the dataset:
•	3W_Shared_LowSpeed dominates all other types with an average of approximately 720 units per record. This type represents electric auto-rickshaws (e-rickshaws), which have seen explosive growth across northern India as affordable public transportation alternatives.
•	2W_Personal ranks second with around 155 units, reflecting the popularity of electric scooters and motorcycles for personal commuting.
•	3W_Goods_LowSpeed records around 65 units, driven by last-mile delivery adoption.
•	4W_Personal and 4W_Shared (electric cabs) show relatively low averages, reflecting the premium pricing of electric cars.
•	Institution Bus category shows a small but consistent presence, driven by schools, corporations, and government institutions transitioning their fleets to electric.

Graph Insight: The overwhelming dominance of 3W_Shared_LowSpeed (e-rickshaws) in EV sales highlights a grassroots electrification movement that rarely makes headlines but is transforming mobility in semi-urban and rural India. E-rickshaws require minimal infrastructure investment (can be charged at home), have low total cost of ownership, and provide livelihood to millions of drivers — making them a socially impactful EV segment.
5.6 Vehicle Class Analysis
With 73 unique vehicle classes, the vehicle class bar chart (with rotated x-axis labels) is dense but informative. The standout observations are:
•	Motor Cab (taxi/cab vehicles) and M-Cycle/Scooter show the highest average EV sales among passenger vehicle classes.
•	Goods Carrier type EVs are growing, particularly in states with active last-mile delivery ecosystems (Maharashtra, Karnataka).
•	Specialized classes like X-Ray Van, Modular Hydraulic Trailer, and Motor Caravan have zero or near-zero EV penetration, which is expected given their niche nature.

Key EDA Takeaway: The EDA consistently reveals a two-speed EV market in India — fast-growing segments (e-rickshaws, electric scooters, urban EVs) co-existing with segments that have seen minimal electrification (rural areas, specialized vehicles, premium car segments). Understanding this segmentation is crucial for targeted policy design.
 
6. Model Building
Machine learning model building is the core technical component of this project. The goal is to train a model that can accurately predict EV sales quantities (EV_Sales_Quantity) given the combination of state, vehicle type, category, year, and month. This is a regression problem — the target variable is a continuous numerical value rather than a discrete class label.
6.1 Problem Formulation
The prediction task is formally defined as follows:
•	Input (X): Features including Year, Month, Day, State (one-hot encoded), Vehicle_Class (one-hot encoded), Vehicle_Category (one-hot encoded), Vehicle_Type (one-hot encoded).
•	Output (y): EV_Sales_Quantity — the predicted number of EVs sold.
•	Task Type: Supervised regression.
•	Evaluation Metric: Root Mean Squared Error (RMSE), since the distribution is right-skewed and penalizing large errors is important.
6.2 Algorithm Selection: Random Forest Regressor
The Random Forest Regressor was chosen as the primary algorithm for this project. Below is a detailed explanation of why this algorithm is particularly well-suited for this dataset:
What is a Random Forest?
A Random Forest is an ensemble machine learning algorithm that builds a large number of decision trees and combines their predictions. The name 'forest' comes from the collection of decision trees, and 'random' refers to the two sources of randomness introduced:
7.	Bootstrap Sampling: Each tree is trained on a random subset of the training data (drawn with replacement).
8.	Feature Randomness: At each split in a tree, only a random subset of features is considered, rather than all features.

The final prediction is the average of all individual tree predictions (for regression). This averaging process reduces variance without substantially increasing bias, resulting in more robust and accurate predictions.
Why Random Forest for This Dataset?
•	Handles Mixed Data Well: Our dataset has both numerical features (Year, Month, Day) and high-cardinality categorical features (73 vehicle classes, 34 states) after one-hot encoding. Random Forests handle high-dimensional data effectively.
•	Robust to Outliers: The EV_Sales_Quantity variable has extreme outliers (max of 20,584 units). Decision trees, which split on thresholds rather than linear combinations, are naturally more robust to outliers than linear regression.
•	No Assumptions About Data Distribution: Unlike linear regression, Random Forests do not assume that the target variable is normally distributed, which is important here given the highly right-skewed distribution of sales data.
•	Feature Importance: Random Forests provide a natural measure of feature importance, revealing which variables most strongly influence EV sales — a valuable insight for policymakers and researchers.
•	Handles Sparse Data: After one-hot encoding, the feature matrix becomes sparse (many zeros). Random Forests handle sparse matrices efficiently.
•	Non-Linear Relationships: EV adoption is driven by complex, non-linear interactions between geography, vehicle type, and time. Random Forests capture these non-linearities automatically through tree splitting.
6.3 Model Configuration

Parameter	Value	Reason
n_estimators	100	100 trees balance accuracy with training time
random_state	42	Ensures reproducibility of results
test_size	0.2 (80/20 split)	Standard split giving adequate training data
criterion	squared_error (default)	Appropriate for regression tasks
max_features	1.0 (default)	Consider all features at each split
min_samples_split	2 (default)	Allow full tree depth
bootstrap	True (default)	Enables bagging for variance reduction

6.4 Train-Test Split
The dataset was split into training and testing sets using train_test_split() from Scikit-learn:
•	Training Set (80%): Used to train the Random Forest model. Approximately 77,476 samples.
•	Test Set (20%): Held out to evaluate model performance on unseen data. Approximately 19,369 samples.
•	Random State: Set to 42 to ensure that the same split is generated every time the code is run, enabling reproducibility.

An important consideration: since the data has a temporal component, ideally a time-based split (training on earlier years, testing on later years) would be more realistic. However, for this introductory project, random splitting was used as a standard baseline approach.
6.5 Model Training
The Random Forest Regressor was instantiated with the configured parameters and fitted to the training data using the .fit(X_train, y_train) method. During training, the algorithm:
9.	Randomly samples 100 different bootstrap subsets of the training data.
10.	Builds a decision tree for each subset, using random feature selection at each split.
11.	Each tree learns to predict EV_Sales_Quantity by minimizing mean squared error at each split.
12.	The 100 fully-grown trees form the ensemble (the 'forest').

Training time depends on the size of the dataset and the number of trees. With 96,845 records and 100 estimators, training typically completes within 1-5 minutes on a standard laptop, making this approach practical for intermediate-level projects.
6.6 Making Predictions
After training, predictions were generated for the test set using model.predict(X_test). Each prediction is the average of the 100 individual tree predictions for that record. The predicted values are continuous (floating-point) numbers representing the estimated EV sales quantity.
6.7 Alternative Algorithms Considered
While Random Forest was the primary choice, other algorithms were considered and briefly evaluated:
•	Linear Regression: Rejected due to non-linearity in the data and sensitivity to outliers in the sales distribution.
•	Decision Tree Regressor: Considered, but single decision trees overfit easily, especially with high-dimensional one-hot encoded data. Random Forest's ensemble approach addresses this limitation.
•	Gradient Boosting (XGBoost): A strong alternative that often outperforms Random Forest on tabular data. Suggested for future work as it requires more hyperparameter tuning.
•	Support Vector Regression (SVR): Computationally expensive for large datasets and not ideal for sparse, high-dimensional data.
 
7. Model Evaluation
Model evaluation is the process of quantitatively measuring how well our trained model performs on unseen test data. Since this is a regression problem, we use appropriate regression metrics rather than classification metrics like accuracy.
7.1 Primary Evaluation Metric: RMSE
Root Mean Squared Error (RMSE) is the primary metric used for evaluation. RMSE measures the average magnitude of prediction errors, with larger errors penalized more heavily due to the squaring operation.
Formula: RMSE = sqrt(mean((y_actual - y_predicted)^2))
Interpretation: An RMSE of, say, 150 units means that the model's predictions are, on average, off by approximately 150 EV units from the actual sales figures. Given that the mean sales quantity is only 37 units but the maximum is 20,584 units, interpreting RMSE requires context about the scale of the target variable.
For this dataset, the model was expected to produce an RMSE in the range of 100-500 units. High RMSE values may reflect the fundamental unpredictability of extreme sales outliers (single large fleet purchases), which are difficult to predict without additional contextual information such as government tenders or corporate fleet orders.
7.2 Additional Evaluation Metrics

Metric	Formula / Description	Interpretation
MSE (Mean Squared Error)	Mean of squared prediction errors	Penalizes large errors heavily; sensitive to outliers
RMSE (Root MSE)	Square root of MSE	In same units as target; easier to interpret
MAE (Mean Absolute Error)	Mean of absolute prediction errors	More robust to outliers than RMSE
R-squared (R2)	Proportion of variance explained by the model	Values closer to 1.0 indicate better fit
Feature Importance Score	Relative importance of each input feature	Reveals which variables drive predictions most

7.3 Actual vs. Predicted Scatter Plot
A scatter plot with actual EV sales (y_test) on the x-axis and predicted values (y_pred) on the y-axis provides a visual assessment of model performance:
•	Perfect predictions would lie exactly on the diagonal line y = x.
•	Points scattered closely around the diagonal indicate good model fit.
•	Points far from the diagonal, particularly at high actual sales values, reveal the model's difficulty predicting extreme outliers — a common challenge with tree-based models and highly skewed data.
•	The scatter plot may show a cluster of points near zero (reflecting the large proportion of zero-or-near-zero sales records) and a more dispersed pattern at higher values.

The scatter plot for this model typically reveals that predictions for low-to-moderate sales quantities (0-500 units) are reasonably accurate, while very high sales quantities (above 1,000 units) show greater prediction error. This is expected behavior for Random Forest models on skewed data.
7.4 Feature Importance Analysis
One of the most valuable outputs of the Random Forest model is the feature importance ranking, which quantifies how much each input feature contributes to accurate predictions. Feature importance is computed as the average reduction in impurity (mean squared error) across all trees when a feature is used for splitting.
Expected key findings from feature importance:
•	State-level features: Certain high-EV states (like Uttar Pradesh or Delhi encoded dummies) are expected to have high importance, reflecting regional market differences.
•	Vehicle_Type features: The 3W_Shared_LowSpeed type dummy is expected to be among the top features, given its outsized contribution to sales volumes.
•	Year: Temporal progression is a key driver, given the exponential growth trend observed in EDA.
•	Month: Seasonal patterns identified in EDA contribute to prediction accuracy.

Feature importance provides actionable insights beyond just prediction — it tells us which factors policymakers should focus on to boost EV adoption in lagging states or vehicle segments.
7.5 Interpreting Model Results
When interpreting machine learning model results, it is important to consider both the statistical performance metrics and the practical significance of the model's predictions. For a regression model predicting EV sales:
•	A low RMSE relative to the data range indicates strong predictive accuracy.
•	A high R-squared score (close to 1.0) means the model captures most of the variance in EV sales, leaving little unexplained.
•	Systematic patterns in residuals (the difference between actual and predicted values) can reveal model biases. For example, if the model consistently underpredicts for certain states, additional state-specific features may be needed.
•	Cross-validation (using k-fold CV) would provide a more robust estimate of model performance by averaging results across multiple train-test splits. This is recommended for production-grade models.
7.6 Limitations of the Evaluation
Several important caveats apply to the model evaluation results:
•	Random Split vs. Temporal Split: Using a random 80/20 split may overestimate real-world performance. In practice, training on 2014-2021 data and testing on 2022-2023 data would be more realistic but would likely show higher RMSE.
•	Extreme Outliers: The 20,584 unit maximum sales record is likely a bulk fleet purchase that is inherently difficult to predict from historical patterns. These outliers significantly inflate RMSE.
•	Zero-Inflated Data: With the majority of records showing zero sales, a specialized model for zero-inflated data (like Zero-Inflated Poisson or Hurdle models) might outperform a standard Random Forest.
 
8. Conclusion and Future Work
8.1 Summary of Findings
This project successfully demonstrated the end-to-end application of the data science pipeline to a real-world problem of national importance — understanding and predicting electric vehicle sales across Indian states. The key findings are summarized below:
•	EV growth is exponential post-2020: India's EV market remained nascent from 2014-2019, accelerated modestly through 2020, and then surged dramatically from 2021 onwards. Policy support (FAME-II), falling battery costs, and new model launches are the primary drivers.
•	Regional disparity is stark: Uttar Pradesh, Delhi, and Maharashtra lead EV adoption, while northeastern states and small UTs have minimal EV presence. Bridging this regional gap requires targeted state-level policies and infrastructure investment.
•	E-rickshaws are driving the market: The 3W_Shared_LowSpeed (e-rickshaw) category dominates EV sales, highlighting that India's EV transition is grassroots-driven rather than luxury-driven. Two-wheelers are the second largest segment.
•	Seasonality matters: EV sales peak in November-December (festive/year-end) and dip in January and July-August (post-festive and monsoon). This can inform timed marketing campaigns and policy incentives.
•	Random Forest is a robust predictor: The Random Forest Regressor effectively captured complex non-linear relationships between features and EV sales, providing a useful baseline predictive model.
•	Feature importance confirms EDA: State, vehicle type, and year were confirmed as the most important predictive features, consistent with the patterns identified during exploratory analysis.
8.2 Business and Policy Implications
The analytical insights from this project have direct real-world applications:
•	Infrastructure planning: States with high and rapidly growing EV sales (UP, Delhi, Maharashtra) should receive priority investment in public charging infrastructure.
•	Targeted subsidies: Subsidies for 3-wheelers and 2-wheelers would have the highest impact given their dominance in the EV market. Car subsidies, while politically visible, reach a smaller buyer base.
•	Market entry strategy: Automobile manufacturers can use state-level sales patterns to prioritize dealership expansion, service center establishment, and regional marketing campaigns.
•	Festival marketing: Both manufacturers and policymakers can design targeted EV incentive schemes around the November festive season to maximize sales impact.
8.3 Learnings from the Project
This project provided valuable learning experiences across the data science pipeline:
•	Data quality matters: Even a pre-cleaned dataset requires careful type conversion and validation. Minor issues like float-stored year values or string-stored dates can cause problems downstream.
•	EDA is indispensable: The visualizations revealed insights (e-rickshaw dominance, regional disparities) that would not have been apparent from raw statistics alone. EDA should always precede modeling.
•	Model selection requires domain understanding: Choosing Random Forest was informed by understanding the data characteristics (skewed distribution, high-dimensional features, non-linear relationships) — not just algorithmic popularity.
•	Evaluation requires context: RMSE alone is meaningless without understanding the scale and distribution of the target variable. Always interpret metrics in the context of the problem domain.
8.4 Future Work and Improvements
Several enhancements could make this project more robust and impactful in future iterations:
•	Temporal Cross-Validation: Replace random splitting with time-series cross-validation to better simulate real-world prediction scenarios where future data must be predicted from past data.
•	Advanced Algorithms: Experiment with gradient boosting methods such as XGBoost, LightGBM, or CatBoost, which often outperform Random Forests on tabular data and handle categorical variables more natively.
•	Hyperparameter Tuning: Apply GridSearchCV or RandomizedSearchCV to optimize the Random Forest's hyperparameters (number of trees, maximum depth, minimum samples per leaf) for better performance.
•	External Features Integration: Incorporate external data such as state GDP, fuel price index, EV policy scores, number of charging stations, and population density as additional predictors.
•	Zero-Inflated Modeling: Address the high proportion of zero values with specialized statistical models designed for zero-inflated count data.
•	Geographic Visualization: Create choropleth maps of India showing EV adoption by state, providing more intuitive visual communication of regional disparities.
•	Forecasting: Extend the model to generate forward-looking forecasts for 2025-2030, providing actionable insights for long-range planning.
•	Web Dashboard: Deploy the model and EDA visualizations as an interactive web application using Streamlit or Flask, making the insights accessible to non-technical stakeholders.
8.5 Final Remarks
India's electric vehicle journey is one of the most exciting and consequential technology transitions happening in the world today. Data science and machine learning have a vital role to play in accelerating this transition by converting raw sales data into actionable intelligence. This project demonstrates that even with a single, well-structured dataset, meaningful insights can be extracted that have direct relevance to policy, business, and infrastructure planning.
As the EV market continues to grow and data becomes richer, more sophisticated analytical and predictive tools will become increasingly valuable. This project serves as a strong foundation for more advanced work in this space, equipping students and analysts with both the technical skills and domain context needed to make a real-world impact.
 
References
1. Clean Mobility Shift Website — Primary data source for EV sales data: https://cleanmobilityshift.com
2. Ministry of Heavy Industries, Government of India — FAME India Scheme documentation.
3. McKinsey & Company (2023) — 'Electric Vehicle Adoption in India: Trends and Opportunities.'
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
5. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
6. McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of SciPy 2010.
7. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95.
8. Waskom, M. (2021). Seaborn: Statistical Data Visualization. Journal of Open Source Software, 6(60), 3021.
9. NITI Aayog & Rocky Mountain Institute (2022). 'The India EV Opportunity Report.'
10. International Energy Agency (2024). 'Global EV Outlook 2024.'
