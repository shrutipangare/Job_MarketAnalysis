# Job Market Analysis

## Overview
The job market, including engineering, finance, and corporate sectors, has been affected by economic shifts, automation, and changing workplace dynamics.

## Project Description
This project tackles that challenge by building a data-driven Job-Market Intelligence Engine that:
- Classifies job titles into meaningful industry categories using supervised learning to impose structure on the fragmented language of online postings
- Maps workforce skills to career opportunities by clustering jobs on skill vectors, revealing natural groupings of roles that demand similar capabilities
- Quantifies skill gaps and generates personalized career recommendations through predictive models that compare current talent supply to forecasted demand across industries

By performing classification, clustering, and forecasting, the project provides actionable insights for:
- Job seekers (targeted upskilling paths)
- Employers (evidence-based talent strategies)
- Policymakers (data-backed workforce development)

## Jupyter Notebook Overview
  * Job Classifcation
- Cleaned and Merged Data Set is Loaded
- Text Cleaning: Normalized noisy job titles by lowercasing, removing fluff terms, and standardizing format.
- Spark ML Pipeline: Tokenized titles, removed stopwords, and extracted features using HashingTF + IDF. Trained a baseline logistic regression model.
- Manual Label Seeding: Labeled ~70 common titles across industries (Tech, Healthcare, etc.) to bootstrap training.
- Full Dataset Inference: Applied the model to classify all job titles into industries.
- Refined with scikit-learn: Re-trained using TfidfVectorizer and LogisticRegression for better accuracy on short text.
- Visualization: Showed job volume by predicted industry using bar plots.



  * Skill Gap Analysis and Career Recommendations
- Skill Clustering: Used KMeans on binary skill indicators (has_ columns) to group users with similar abilities.
- Cluster Interpretation: Identified dominant skills per cluster to describe each group meaningfully.
- Sector Clustering: Grouped job sectors by average skill demand using Spark + Pandas + KMeans.
- Skill Gap Analysis: Compared a user's skill profile to each cluster center using cosine similarity to highlight missing skills.
- Career Transition Paths: Suggested target clusters the user can move toward and the top skills needed for that transition.

## Data Sources

### Job Description Dataset
- **URL:** [https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset)
- **Size:** 1.74 GB
- **Records:** 1,615,940

### LinkedIn Jobs and Skills 2024
- **URL:** [https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024)
- **Size:** 6.19 GB
- **Records:** 1.3 million

### Files:
- Data_preparation.ipynb - Loading and Data Processing
- Basic_ExploratoryAnalysis.ipynb - Performed some data Analysis
- Job_classification.ipynb, Skill and Career Rec.ipynb - ML models
- job_analysis_dashboard.twb- Tableau Dashboard


## Tech Stack
Google Colab/Jupyter Notebook, Python, Pyspark, PySQL, Spark, MLlib, Pandas, Numpy, Matplotlib, Seaborn, Tableau

## Team Members
- Ilka Jean (ifj2007)
- Neha Nainan (nan6504)
- Shruti Pangare (stp8232)
