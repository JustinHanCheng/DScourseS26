-- PS3 SQL script for Florida insurance sample
-- (a) Read in the CSV file
.open PS3_Cheng.db
DROP TABLE IF EXISTS florida_insurance;
CREATE TABLE florida_insurance (
  policyID TEXT,
  statecode TEXT,
  county TEXT,
  eq_site_limit REAL,
  hu_site_limit REAL,
  fl_site_limit REAL,
  fr_site_limit REAL,
  tiv_2011 REAL,
  tiv_2012 REAL,
  eq_site_deductible REAL,
  hu_site_deductible REAL,
  fl_site_deductible REAL,
  fr_site_deductible REAL,
  point_latitude REAL,
  point_longitude REAL,
  line TEXT,
  construction TEXT,
  point_granularity INTEGER
);
.mode csv
.import FL_insurance_sample.csv florida_insurance
DELETE FROM florida_insurance WHERE policyID = 'policyID';

-- (b) Print first 10 rows
SELECT * FROM florida_insurance LIMIT 10;

-- (c) List unique counties
SELECT DISTINCT county FROM florida_insurance ORDER BY county;

-- (d) Average property appreciation from 2011 to 2012
SELECT AVG(tiv_2012 - tiv_2011) AS avg_property_appreciation
FROM florida_insurance;

-- (e) Frequency table for construction and fraction of total
SELECT
  construction,
  COUNT(*) AS n,
  ROUND(1.0 * COUNT(*) / (SELECT COUNT(*) FROM florida_insurance), 4) AS fraction
FROM florida_insurance
GROUP BY construction
ORDER BY n DESC;
