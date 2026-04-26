# Business Case Analysis

### Scenario: Promotion Effectiveness at a Fashion Retail Chain
A fashion retailer operates 50 stores across urban, semi-urban, and rural locations. Each month, the marketing team runs one of five promotions: Flat Discount, BOGO (Buy-One-Get-One), Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. Stores vary in size, monthly footfall, local competition density, and customer demographics. The company wants to determine which promotion should be deployed in each store each month to maximise the number of items sold.

---
### Problem Formulation
**B1.a:** It is  a **Multi-Class Classifiction** as the model needs to predict one of discrete promotion categories with no numeric ordering between them making regression inapporiate.

Input Features `store_Location`, `year`, `month`,  `stores_size`, `monthly_footfall`, `competition_density`, and `population`.  

Target label `promotion`  

**B1.b:**   
Revenue is influenced by product price, not just promotion effectiveness. A store selling fewer but expensive items can show high revenue even if the promotion performed poorly in terms of customer reach.  

Items sold directly measures how many customers responded to the promotion, regardless of what price the item carried. It isolates the promotion's effect.  

This illustrates the principle of metric-goal alignment the target variable must directly measure what the business wants to optimise. When a metric is influenced by factors outside the intervention (like price), the model learns to optimise the wrong thing.  

**B1.c:**  
A single global model assumes all 50 stores share the same patterns, but stores in urban, semi-urban, and rural locations attract fundamentally different customers with different buying behaviours. This causes the model to average out local patterns, making it inaccurate for any specific store type.

A better strategy is store clustering grouping stores by shared characteristics such as location type, store size, footfall range, and customer demographics. A separate model is then trained for each cluster.  

This balances two extremes: a global model is too general, while 50 individual models would have too little data per store to train reliably. Clustered models capture local behaviour patterns while still having sufficient training data within each group.

### Data and EDA Strategy
**B2.a:**  
`Grain`: Each row in the final dataset represents one store in one month capturing that store's features, the promotion run, and the resulting sales volume.

`Transactions table`: Aggregate by store_id, year, month computing total items sold, total revenue, and transaction count. This collapses hundreds of transactions into one summary row per store per month.

`Store attributes table`: Already at store grain join directly on store_id.
Promotion details table: Join on store_id + year + month to bring in which promotion was run.  

`Calendar table`: Aggregate daily flags to monthly level count number of weekend days and festival days per month. Join on year + month.  

All four tables are linked through the composite key of `store_id + year + month`.

**B2.b:**  
1. Data Quality Check: Inspect all columns for missing values. Random missingness → impute with median/mode. Patterned missingness → investigate cause and add a binary flag column. Outliers in items_sold or footfall → check if genuine or data errors.


2. Items Sold Distribution (Histogram): Check if target variable is skewed. A right-skewed distribution may require log transformation before modelling.


3. Average Items Sold by Location Type (Bar Chart): Check if urban/rural stores consistently differ in sales volume. If yes, confirms location_type is a strong feature and validates the clustering strategy.


4. Monthly Sales Trend (Line Chart): Identify seasonal peaks and dips. If strong seasonality exists, month becomes a critical feature and festive-month flags should be engineered.


5. Promotion × Location Heatmap: Average items sold for each promotion-location combination. Reveals interaction effects if certain promotions only work in certain locations, create an interaction feature.


6. Footfall vs Items Sold Scatter Plot (coloured by promotion): Check correlation between footfall and sales, and whether it differs by promotion type. Guides feature importance decisions.


**B2.c:**  
With 80% no-promotion data, the model learns to predict "no promotion" for almost everything and still achieves 80% accuracy making accuracy a completely misleading metric.  

Missing a promotion when one was needed (false negative) costs the business through lost sales and reduced revenue. Running an unnecessary promotion (false positive) is a smaller cost. Therefore Recall which measures how well we catch all true promotion cases is the right evaluation metric.

To fix this evaluate using Recall and F1-Score instead of accuracy

### Model Evaluation and Deployment
**B3.a:**  
`Train-Test Split:`The data should be split by time Year 1 and Year 2 used for training, Year 3 used for testing. This mirrors real deployment conditions where the model always predicts future months using only past data. Time must always move forward.  

`Random Split is Inappropriate:`
A random split may include future data (Year 3) in the training set and past data (Year 1) in the test set. This causes data leakage the model sees future patterns during training, making test performance look artificially strong. In real deployment, future data never exists, so the model would fail in practice.  

`Evaluation Metrics:`Accuracy is misleading here with 80% no-promotion data, a model that always predicts "no promotion" scores 80% accuracy while being completely useless.

*Recall* is the priority metric it measures how many true promotion opportunities the model correctly identified. Missing a promotion when one was needed costs the business through lost sales and reduced revenue.  

*F1-Score* balances Recall with Precision ensuring the model doesn't over-recommend promotions either, which would waste marketing budget.

**B3.b:**  
December → Loyalty Points Bonus:
The model detected high festive footfall and gifting season. Customers are already motivated to shop and likely to return multiple times. Loyalty Points rewards repeat visits maximising long-term sales during peak season.  

March → Flat Discount:
No festive flags, normal footfall. Customers need immediate motivation to purchase. Flat Discount provides instant visible savings the simplest trigger for impulse buying in off-peak months.  

Communication to Marketing Team:
"In December, customers are already in a shopping mood and will return naturally so rewarding loyalty makes sense. In March, customers need a direct reason to walk in an instant discount is the strongest motivator. The model picked up these seasonal patterns from 3 years of historical data."

**B3.c:**  
Saving the Model:
After training, the model is serialized into a file using joblib or pickle preserving all learned patterns. This file is stored on a server so it can be loaded instantly each month without retraining.


Monthly Data Pipeline:
At the start of each month, an automated pipeline pulls the previous month's data from the data warehouse. The data is cleaned and aggregated using identical steps to training producing one row per store. This is fed into the saved model, generating promotion recommendations for all 50 stores, which are delivered to the marketing team dashboard.


Monitoring:
Two types of drift are monitored. Data drift checking whether new monthly inputs have shifted significantly from training data distributions, such as footfall ranges or demographic patterns. Performance drift comparing model recommendations against actual sales outcomes each month, tracking F1-Score over time. If F1-Score drops below a defined threshold (e.g. 70%), an alert is triggered and the model is retrained on the most recent data.