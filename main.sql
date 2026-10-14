CREATE DATABASE customer_segmentation;
USE customer_segmentation;

CREATE TABLE transactions (
    InvoiceNo VARCHAR(20),
    StockCode VARCHAR(20),
    Description TEXT,
    Quantity INT,
    InvoiceDate DATETIME,
    UnitPrice DECIMAL(10, 2),
    CustomerID BIGINT,
    Country VARCHAR(50)
);

ALTER TABLE transactions MODIFY COLUMN CustomerID BIGINT NULL;


SHOW VARIABLES LIKE 'secure_file_priv';

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/invoice_data_clean_2.csv'
INTO TABLE transactions
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"' 
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(InvoiceNo, StockCode, Description, Quantity, @InvoiceDate, UnitPrice, @CustomerID, Country)
SET
  InvoiceDate = STR_TO_DATE(@InvoiceDate, '%d-%m-%Y %H:%i'),
  CustomerID = CAST(@CustomerID AS UNSIGNED);


SELECT * FROM transactions



-- Remove canceled invoices (those starting with 'C')
DELETE FROM transactions
WHERE InvoiceNo LIKE 'C%'
LIMIT 1000;


-- Remove rows with zero or negative quantity or price
DELETE FROM transactions
WHERE Quantity <= 0 OR UnitPrice <= 0
ORDER BY InvoiceNo
LIMIT 1000;

CREATE TABLE RFM_Data AS
SELECT 
    CustomerID,
    DATEDIFF('2011-12-10', MAX(InvoiceDate)) AS Recency,
    COUNT(DISTINCT InvoiceNo) AS Frequency,
    ROUND(SUM(Quantity * UnitPrice), 2) AS Monetary
FROM transactions
WHERE CustomerID is not NULL 
GROUP BY CustomerID;

SELECt * FROM RFM_Data ;
SELECT COUNT(*) FROM RFM_Data;


SELECT COUNT(*) AS total_rows,
       COUNT(CustomerID) AS customer_id_present,
       COUNT(DISTINCT CustomerID) AS distinct_customers
FROM transactions;


SELECT COUNT(*) FROM transactions;


SELECT CustomerID, InvoiceDate
FROM transactions
LIMIT 10;
