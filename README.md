# Using Unsupervised Learning Clustering algorithms in EDA of a real-world healthcare dataset
## WiDS Datathon 2020: Challenge focused on social impact
*Karen Matthys, Marzyeh Ghassemi, Meredith Lee, NehaGoel, Sharada Kalanidhi, sumalaika. (2020). WiDS Datathon 2020. Kaggle. https://kaggle.com/competitions/widsdatathon2020*

> The WiDS Datathon 2020 focuses on patient health through data from MIT’s GOSSIS (Global Open Source Severity of Illness Score) initiative. MIT's GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States.

The task of the WiDS challenge (not to be confused with the purpose of this notebook) is to create a model that uses data from the first 24 hours of intensive care to predict patient survival ('hospital_death')

## The Dataset
The [WiDS datathon 2020 training_v2 dataset](https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fc%2Fwidsdatathon2020%2Fdata%3Fselect%3Dtraining_v2.csv) has **91,173 records** and **186 variables**, which belong to following categories:
| Category | Description |
| :-- | :-- |
| Identifier | Unique identifier associated with a patient or hospital |
| Demographic |  Demographic info of patient on admission |
| APACHE covariate |  APACHE diagnosis code for ICU admission; patient measurements taken during the first 24 hours of unit admission |
| Vitals (d1) | Patient vital measurements (xxx) taken during the first 24 hours of their unit stay |
| Vitals (h1) | Lowest and highest vital measurements (xxx) taken during the first hour of their unit stay |
| Labs |  Lowest and highest lab measurements (yyy) taken during the first 24 hours of their unit stay |
| Labs blood gas | Highest and lowest arterial measurements for the patient during the first hour (h1) or first 24 hours (d1) of their unit stay |
| APACHE comorbidity | Binary (0 or 1); Whether the patient has been diagnosed with this comorbidity |

## Purpose of this notebook
### Supervised vs Unsupervised Learning
| | Supervised Learning | Unsupervised Learning|
| :-- | :-- | :-- |
| Data | Labeled | Unlabeled|
| Algorithm | Trained on a dataset with known inputs and outputs | Given a dataset with no predetermined outcomes |
| Goal | Learn a function that can accurately map new inputs to their corresponding outputs | Uncover the inherent structure and patterns within the data |
| Interpretability | Generally more straightforward and  easier, as the algorithm is trained to optimize a specific objective | More complex and may require domain knowledge to interpret the results, as the algorithm is not guided by a specific target variable |

The purpose of this notebook is to **implement Unsupervised Learning methodologies** I've learned thus far. In this notebook, I will **explore the data and investigate potential underlying relationships, patterns, and insights** that may exist within the dataset. 

### Methods: Exploratory Data Analysis, Dimensionality Reduction, Unsupervised Learning Clustering algorithms
1. Perform careful, thoughtful, and successful EDA to produce a "clean" training dataset
   * A reproducible process is detailed and established in my [Capstone II notebook](https://colab.research.google.com/drive/1MHqFPqzG63W8ejOyRyflMdMH8veaR10m?usp=sharing)
2. Evaluate dimensionality reduction techniques and utilize the technique best suited to the clean training dataset
   * Dimensionality reduction techniques: PCA, t-SNE, UMAP
3. Build several clustering models and evaluate their performance
   * Models: K-means, Hierarchical clustering, DBSCAN, GMM
   * Evaluation metrics: Silhouette Coefficient, Davies-Bouldin Index, Calinski-Harabasz Index
       * *Please note that ARI Score is not included. While the Ground Truth is present in the dataset, it will not be used to guide the exploration of this data using Unsupervised Learning Clustering algorithms*

### Using Unsupervised Learning to explore a real-world healthcare dataset

I learned that *Unsupervised learning in healthcare is [semipopular in healthcare](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8822225/) because it is considered "usually purposeful in data analysis, stratification, and reduction rather than prediction."* ML in medicine and healthcare is a complex problem being addressed actively by many brilliant minds, so the objective in this notebook isn't to develop a model that will make it to the bedside, but to **use unsupervised learning clustering algorithms to identify underlying relationships in this dataset and group them by similarities**.

For example, *will the features within a cluster provide a meaningful and useful "snapshot" of patient?* Will a cluster contain a mix of variables from different categories (e.g., patient demographics, critical vital measurements and lab measurements) that help characterize the patient meaningfully? Or will the clusters simply reflect the categories of the data (i.e., a cluster of demographic information, a separate cluster of lab measurements, a separate cluster of vital measurements? etc.)? *The goal is to see clusters and understand relationships in the data, so I need to be able to interpret cluster information in terms of individual features (not just components).*

## Exploratory Data Analysis
1. Handle Categorical Variables
   * Label encoding
2. Deal with outliers
   * Log transformation of continous, non-binary variables
3. Scale the continuous variables
   * All dimensionality reduction techniques are sensitive to the scale of the data
   * StandardScaler()
4. Normalize the log-transformed and standardized continuous variables
   * Yeo Johnson Transformation
### A clean dataframe suitable for modeling 
* Lacks missing or NaN values and records
* Has new label-encoded features that were generated from categorical variables
* Has log-transformed continuous features (where applicable)
* Has scaled and normalized continuous features
The clean, post-data cleaning and feature engineering dataframe has **66,631 records** and **67 features/variables**

## Dimensionality reduction techniques

* Principal Components Analysis (PCA)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)
* Uniform Manifold Approximation and Projection (UMAP)

### Comparing PCA, t-SNE, and UMAP
| | PCA | t-SNE | UMAP |
| :-- | :-- | :-- | :-- |
| Linear or Non-linear | Linear $^*$ | Non-linear | Non-linear |
| Preservation of Global or Local Structure | Global | Local | Both |
| Visualization or Feature Extraction | Both $^*$ | Visualization $^*$$^*$ | Both AND $^*$$^*$$^*$ |
| Interpretability | PCs are linear combinations of original features; relatively more interpretable | More difficult; doesn't have a direct correspondence to original features | More difficult to interpret, but can capture more nuanced relationships in the data |
| Scalability | More scalable than t-SNE | Computationally expensive for large (>10K_ and dimensionally large (>50) datasets | More scalable than t-SNE|
| Speed | Fastest | Slowest | Second-fastest|
$^*$ PCA has a number of assumptions (see below for more details)
$^*$$^*$ t-SNE is also commonly used in EDA
$^*$$^*$$^*$ UMAP is also particularly useful in EDA, clustering and classification tasks, anomaly detection, and preprocessing for machine learning models

### Apply the UMAP dimensionality reduction technique

The shape of the clean_train_df is (66631, 67). The shape of this dataset precludes t-SNE (>10K records and >50 features). Ultimately, the variables in this dataset have complex, nonlinear relationships, so a nonlinear dimensionality reduction technique is appropriate for this dataset. I will proceed with UMAP and use RandomizedSearchCV(0 to get the best parameters for UMAP.

## Clustering algorithms in EDA
K-means, Hierarchical Clustering, DBSCAN, and GMM are the major clustering algorithms. K-means and GMM were implemented, but the UMAP-reduced data do not meet the assumptions of these algorithms, so the results are not shown here. 

### Hierarchical Clustering (HC) produced a three-cluster solution
#### Dendrogram using the Complete linkage method
![image](https://github.com/user-attachments/assets/2ce19407-7847-4e7d-a049-9666d92e38e6)
#### Dendrogram using the Average linkage method
![image](https://github.com/user-attachments/assets/45f9276a-93e3-41a2-b097-28bae566fb6d)
#### Dendrogram using the Ward linkage method
![image](https://github.com/user-attachments/assets/8b7d4505-c919-4b28-a59d-94ae609437d8)

| Silhouette Score | Davies-Bouldin Index (DBI) | Calinski-Harabasz Index (CHI) |
|  :--:  | :--: | :--: |
|  0.811898946762085 | 0.2525101717357637 | 615542.1262766493 |

### Density-Based Spatial Clustering (DBSCAN) produced a five-cluster solution
![image](https://github.com/user-attachments/assets/3d57ca51-4ee9-45db-90b4-126676f86a7f)
| min_samples | Number of Clusters | Silhouette Coefficient | DBI | CHI |
| :--: | :--: | :--: | :--: | :--: |
| 15 | **5** | 0.5997769236564636 | 1.5165248590206601 | 270235.9889261402 |


## Centroid Analysis per Cluster
Using ground truth to bring the data full circle to survival and death rates.

**Ground truth: hospital_death**
* *0*: Surival; 92.5%
* *1*: Death; 7.5%

### HC Clusters: The top 20 features in each cluster
| Cluster | 0 | 1 | 2 |
| :-- | :--: | :--: | :--: |
| Value Counts | 31812 | 8876 | 25943 |
| Features |
| 1 |apache_2_bodysystem         |  apache_3j_bodysystem        |  apache_3j_bodysystem        |
| 2 | apache_3j_bodysystem        |  apache_2_bodysystem         |  apache_2_bodysystem         |
| 3 |icu_type                    |  apache_2_diagnosis          |  icu_type                    |
| 4 |apache_3j_diagnosis         |  h1_sysbp_min                |  apache_3j_diagnosis         |
| 5 |apache_2_diagnosis          |  h1_sysbp_max                |  gcs_verbal_apache           |
| 6 |age                         |  d1_sysbp_noninvasive_min    |  gcs_eyes_apache             |
| 7 |gcs_eyes_apache             |  d1_sysbp_min                |  age                         |
| 8 |gcs_verbal_apache           |  h1_heartrate_min            |  d1_temp_max                 |
| 9 |h1_sysbp_min                |  h1_diasbp_min               |  h1_heartrate_min            |
| 10 |h1_diasbp_min               |  h1_resprate_min             |  gcs_motor_apache            |
| 11 |h1_sysbp_max                |  h1_heartrate_max            |  h1_heartrate_max            |
| 12 |d1_sysbp_noninvasive_min    |  d1_mbp_noninvasive_min      |  d1_heartrate_max            |
| 13 |d1_sysbp_min                |  d1_mbp_min                  |  d1_heartrate_min            |
| 14 |h1_diasbp_max               |  h1_resprate_max             |  heart_rate_apache           |
| 15 |d1_diasbp_min               |  d1_heartrate_min            |  icu_admit_source            |
| 16 |d1_diasbp_noninvasive_min   |  d1_heartrate_max            |  temp_apache                 |
| 17 |d1_mbp_noninvasive_min      |  d1_diasbp_noninvasive_min   |  h1_resprate_min             |
| 18 |d1_mbp_min                  |  h1_diasbp_max               |  d1_temp_min                 |
| 19 |gcs_motor_apache            |  d1_diasbp_min               |  h1_resprate_max             |
| 20 |d1_diasbp_max               |  heart_rate_apache           |  d1_resprate_min             |

<img width="567" alt="Screenshot 2024-11-14 at 10 31 07 AM" src="https://github.com/user-attachments/assets/501a56cc-4dca-4391-a15a-8fc4330d9573">

### DBSCAN Clusters: The top 20 features in each cluster

| Cluster | -1 | 0 | 1 | 2 | 3 |
| :-- | :--: | :--: | :--: | :--: | :--: |
| Value Counts | 2906 | 8872 | 25052 | 25185 | 4616 |
| Features |
| 1 | apache_2_bodysystem         | apache_3j_bodysystem        | apache_3j_bodysystem        |  apache_3j_bodysystem        | apache_2_bodysystem         |
| 2 | apache_3j_bodysystem        | apache_2_bodysystem         | apache_2_bodysystem         |  apache_2_bodysystem         | apache_3j_bodysystem        |
| 3 | apache_2_diagnosis          | apache_2_diagnosis          | icu_type                    |  icu_type                    | apache_3j_diagnosis         |
| 4 | icu_type                    | h1_sysbp_min                | age                         |  apache_3j_diagnosis         | apache_2_diagnosis          |
| 5 | gcs_verbal_apache           | h1_sysbp_max                | h1_diasbp_max               | gcs_verbal_apache           | icu_admit_source            |
| 6 | gcs_eyes_apache             | d1_sysbp_noninvasive_min    | h1_diasbp_min               | gcs_eyes_apache             | apache_post_operative       |
| 7 | d1_temp_min                 | d1_sysbp_min                | h1_sysbp_min                | d1_temp_max                 | pre_icu_los_days            |
| 8 | icu_admit_source            | h1_heartrate_min            | h1_sysbp_max                | age                         | elective_surgery            |
| 9 | temp_apache                 | h1_diasbp_min               | d1_diasbp_max               | h1_heartrate_min            | h1_resprate_max             |
| 10 | d1_heartrate_min            | h1_resprate_min             | d1_diasbp_noninvasive_max   | gcs_motor_apache            | h1_resprate_min             |
| 11 | d1_sysbp_min                | h1_heartrate_max            | icu_admit_source            | h1_heartrate_max            | icu_type                    |
| 12 | d1_sysbp_noninvasive_min    | d1_mbp_noninvasive_min      | apache_3j_diagnosis         | d1_heartrate_max            | d1_diasbp_max               |
| 13 | h1_sysbp_min                | d1_mbp_min                  | d1_sysbp_noninvasive_max    | d1_heartrate_min            | d1_diasbp_noninvasive_max   |
| 14 | h1_heartrate_min            | h1_resprate_max             | d1_sysbp_max                | heart_rate_apache           | d1_mbp_noninvasive_max      |
| 15 | heart_rate_apache           | d1_heartrate_min            | d1_sysbp_min                | temp_apache                 | d1_mbp_max                  |
| 16 | map_apache                  | d1_heartrate_max            | d1_mbp_noninvasive_max      | h1_resprate_min             | d1_sysbp_noninvasive_max    |
| 17 | ethnicity                   | d1_diasbp_noninvasive_min   | d1_sysbp_noninvasive_min    | d1_temp_min                 | h1_diasbp_max               |
| 18 | age                         | h1_diasbp_max               | d1_mbp_max                  | icu_admit_source            | d1_sysbp_max                |
| 19 | apache_3j_diagnosis         | d1_diasbp_min               | d1_diasbp_min               | d1_resprate_min             | d1_resprate_min             |
| 20 | h1_diasbp_min               | heart_rate_apache           | gcs_eyes_apache             | h1_resprate_max             | h1_spo2_max |

<img width="559" alt="Screenshot 2024-11-14 at 10 31 56 AM" src="https://github.com/user-attachments/assets/63395ba0-7cc5-4c18-96b2-64c5c1675166">

## Two different clustering methods produced one identical cluster that had the highest mortality rate!

DBSCAN Cluster 0 and HC Cluster 1 were had the highest percentage of death (14.62%). They were nearly identical in value count (8872 vs 8876, respectively) and centroid composition. The *most important features (within the top 10) in these clusters were*
* **Admission diagnosis group for APACHE II & APACHE III**
  * Cardiovascular, Neurologic, Respiratory, Gastrointestinal, Metabolic, Trauma, Undefined, Renal/Genitourinary, Hematological, Musculosckeletal, Sepsis, Gynecological
* **Key vitals measured within the first hour of ICU stay**
  * systolic blood pressure (min & max)
  * heart rate (min)
  * diastolic blood pressure (min)
  * respiratory rate (min)
* **Systolic blood pressure measured within the first 24 hours of ICU stay**
  * systolic blood pressure (min; non-invasive measurement)
  * systolic blood pressure (min)

These results inspire high confidence in the EDA process, cluster formation and subsequent identities of the clusters, and the features identified/selected in this unsupervised learning exercise!


