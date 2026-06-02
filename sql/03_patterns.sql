-- Fraud pattern analysis queries
-- Run against fraud_db with a user that has SELECT on transactions


-- 1. Fraud rate by merchant category
-- Question: which merchant categories concentrate the most fraud?
SELECT category,
       COUNT(*) AS total_transactions,
       SUM(is_fraud) AS fraud_transactions,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY category
ORDER BY fraud_rate_pct DESC;


-- 2. Fraud rate by hour of day
-- Question: at what times does fraud spike?
SELECT EXTRACT(HOUR FROM trans_date_trans_time) AS hora,
       SUM(is_fraud) AS fraud_transactions,
       COUNT(*) AS total_transactions,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY hora
ORDER BY hora;


-- 3. Transaction velocity per card (window function)
-- Question: how many times did the same card transact in the past hour?
SELECT trans_num,
       cc_num,
       trans_date_trans_time,
       amt,
       is_fraud,
       COUNT(*) OVER (
           PARTITION BY cc_num
           ORDER BY unix_time
           RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
       ) AS txns_last_hour
FROM transactions
ORDER BY txns_last_hour DESC;


-- 4. Amount vs. card historical average
-- Question: is this transaction unusually large for this card?
SELECT trans_num,
       cc_num,
       amt,
       is_fraud,
       ROUND(AVG(amt) OVER (PARTITION BY cc_num), 2) AS avg_amt_card,
       ROUND(amt / NULLIF(AVG(amt) OVER (PARTITION BY cc_num), 0), 1) AS times_over_avg
FROM transactions
ORDER BY times_over_avg DESC NULLS LAST;


-- 5. Suspicious transactions view
-- Combines velocity (query 3) and anomalous amount (query 4) into a reusable view
CREATE VIEW suspicious_transactions AS
SELECT *
FROM (
    SELECT trans_num,
           cc_num,
           trans_date_trans_time,
           amt,
           category,
           merchant,
           is_fraud,
           COUNT(*) OVER (
               PARTITION BY cc_num
               ORDER BY unix_time
               RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
           ) AS txns_last_hour,
           ROUND(amt / NULLIF(AVG(amt) OVER (PARTITION BY cc_num), 0), 1) AS times_over_avg
    FROM transactions
) t
WHERE txns_last_hour > 5 OR times_over_avg > 5;
