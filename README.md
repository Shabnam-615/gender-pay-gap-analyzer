Gender Pay Gap Analyzer
A fairness-aware machine learning system that detects and explains gender-based salary disparities using XGBoost, Fairlearn, and SHAP — deployed as an interactive dashboard.
Live app → gender-pay-gap-analyzer.streamlit.app

The finding
The raw pay gap in this dataset is $8,514/year. But the more revealing number comes from the model itself.
An XGBoost model trained to predict salaries — with gender explicitly excluded as a feature — still produced $2,038 more prediction error on women's salaries than men's. The model never saw gender. It learned the bias anyway, from the patterns already present in the data.
This is what systemic bias looks like inside an ML system.

Features

Fairness-aware evaluation using Fairlearn's MetricFrame to measure prediction disparity across gender groups
SHAP explainability with summary and waterfall plots showing exactly which features drive each salary prediction
What-if simulator — adjust job title, department, seniority, age, and education to see predicted salary shift in real time with a full SHAP breakdown
Pay gap heatmap by department showing where disparities are largest
Interactive dashboard built with Streamlit and Plotly


Tech stack
LayerToolsModelingXGBoost, scikit-learnFairnessFairlearnExplainabilitySHAPDashboardStreamlit, Plotly, MatplotlibDataGlassdoor Gender Pay Gap datasetDeploymentStreamlit Cloud

Project structure
gender-pay-gap-analyzer/
├── app.py               # Streamlit dashboard
├── analysis.ipynb       # Exploration, model training, fairness report
├── data/
│   └── Glassdoor Gender Pay Gap.csv
└── requirements.txt

Key results
MetricValueRaw pay gap$8,514/yearModel MAE (overall)$9,942MAE — Female$11,062MAE — Male$9,024Prediction disparity$2,038Top salary driversAge, Seniority, Job Title

Insight
Removing a sensitive attribute from your model does not make it fair. Bias travels through proxy variables — age, seniority, and job title all carry historical inequality encoded in the training data. Fairness requires active measurement, not passive omission.

Run locally
bashgit clone https://github.com/Shabnam-615/gender-pay-gap-analyzer.git
cd gender-pay-gap-analyzer
pip install -r requirements.txt
streamlit run app.py

Author
Shahira Shabnam — LinkedIn · GitHub
