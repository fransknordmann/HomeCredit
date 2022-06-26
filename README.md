# HomeCredit
Config files for my GitHub profile.

"Credit Scoring" project based on Home Credit dataset, available @ Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data
Realized in Python, based on MLFlow (model API), Flask (other web API), Dash (Dashboard)

"API" folder:
from P07_01_* to P07_03_* : data exploration, feature engineering, modelisation tuning, measure
P07_04_* : Creation of MLFlow model and API
P07_05_* : Web APIs, Flask (all data for Dashboard delivered through those APIS, including model score).

"Dashboard" folder:
P7_06_Dashboard.py : HTML Dashboard, Dash

System architecture and workflow:

            Web user
                |
                V
     HTML/Dash P7_06_Dashboard.py
                |
                V
        Flask P7_05_API_Web.py
          ^           ^
          |           |
       *.csv     MLFlow Server
          ^           ^
          |           |
          |     credithome_model
          |           ^
          |           |___ P7_04_Classification_MLFlow_creation.ipynb (model generation)
          |
          |__  P7__03_Classification_Reduced.ipynb (model with reduction to main features)
          
          
Dashboard available at http://oc.14eight.com:8050
Hosting:
Please note there is no HTTPS certificate, please confirm safety of the access to your browser to view the dashboard anyway.
Model generation, MLFlow server, Web API and Dashboard hosted on AWS, EC2 free instance.
Due to limitation of a free instance the model behind the dashboard is limited to 15000 loans (among the original >300K loans)

Modelisation:

P7__03_Classification.ipynb :
 - measure of different model performances, keep GradientBoost.
 - Measure based on Roc Area Under the Curve.
 - Search for False positives and False negative rates (Kfold, ROC)
 - Feature importance (Kfold and feature permutations)
 - register data set removing features with weight = 0 (reduced dataset)

P7__03_Classification_Reduced.ipynb
 - measure performance with reduced dataset
 - check that results are identical as with complete dataset
 - confirm model parameters

P7_04_Classification_MLFlow_creation.ipynb
 - create a sklearn pipeline with model parameters
 - generate MLFlow model credithome_model from pipeline
