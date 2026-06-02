\copy transactions FROM 'C:/Users/pablo/Desktop/fraud-risk-analytics/data/raw/fraudTrain.csv' WITH (FORMAT csv, HEADER true);
\copy transactions FROM 'C:/Users/pablo/Desktop/fraud-risk-analytics/data/raw/fraudTest.csv' WITH (FORMAT csv, HEADER true);
