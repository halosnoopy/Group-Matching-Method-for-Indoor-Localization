# Group-Matching-Method-for-Indoor-Localization
## Overview
The **Group Matching Method (GMM)** is a region-level filtering approach designed to improve indoor localization using Wi-Fi signal fingerprints.  
It works by identifying which Access Points (APs) are the strongest in a given area and using this information to narrow down where a user is likely located.  
By filtering the search space before matching, GMM helps achieve faster and more accurate positioning results.

---

## Motivation
In Wi-Fi-based indoor localization, each location is represented by a set of signal strengths measured from nearby Access Points.  
However, signal values often change due to interference, obstacles, or environmental conditions.  
Although the signal strength can vary, the **set of strongest APs in each area tends to remain stable**.  

This observation forms the key idea behind GMM:  
instead of comparing full signal values, the method focuses on identifying which APs are dominant in the current signal and matches locations that share similar dominant APs.

---

## How It Works

### 1. Offline Preparation
Before the localization process starts, a database of signal fingerprints is prepared:
- Each fingerprint is a record of Wi-Fi signal strengths collected at a known location.
- Missing values (when some APs are not detected) are replaced with a fixed number such as **–100 dBm**.
- For every location, the strongest APs are identified and saved as a simple “top list” that shows which APs dominate in that area.

This creates two datasets:
- **FP dataset:** stores all signal values.
- **RO dataset:** stores which APs are strongest at each location.

These datasets are ready for quick lookup during localization.

---

### 2. Online Localization
When a new signal measurement is received:
1. The system first checks which APs are strongest in the signal.
2. It then looks for reference points in the database that have similar dominant APs.
3. Only these matching points are kept for the next step, reducing unnecessary comparisons.
4. Finally, the user’s position is estimated by averaging the coordinates of the closest matching points.

This process avoids searching the entire database and focuses only on regions that share similar AP patterns, improving both speed and accuracy.
