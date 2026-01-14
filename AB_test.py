"""
What is A/B testing ?

A/B testing is a method to analyse the proportion changes on the feature variants,
using two proportion z test. Which helps to determine the absolute difference and relative difference.
between the old vs new feature change.
In A/B test we calculate the conversion rate for both the features.

What is z test and why using z-test.

z test is the statistical method to evaluate the proportion difference or singnificant change.
using z-test because its used to proportionate the difference between two conversions.

or in z-test specially two samples z-test because we comparing two datasets mean distributions.

"""

## Python- Pandas implementation.
import pandas as pd
import numpy as np
from scipy import stats

#Example dataset
df = pd.DataFrame({
    "Group": ["A", "B"],
    "Visitors": [5000,5000],
    "Conversions":[250,275]
})

df["conversion_rate"] = df["Conversions"]/df["Visitors"]
df
#Calculating the conversion rates uplift/relative difference
cr_A = df.loc[df["Group"]=="A","conversion_rate"].values[0]
cr_B = df.loc[df["Group"] == "B", "conversion_rate"].values[0]

lift = cr_B-cr_A
relative_lift = (lift/cr_A)*100
print(f"Conversion Rate A: {cr_A}, Conversion Rate B: {cr_B}")
print(f"Lift: {lift}, Relative Lift: {relative_lift} %")
#Performing two sample z-test

from statsmodels.stats.proportion import proportions_ztest
conversions = np.array(df["Conversions"])
visitors = np.array(df["Visitors"])
z_stat, p_value = proportions_ztest(conversions,visitors)
print(f"Z-statistic: {z_stat:.2f}, P-value: {p_value:.2f}")
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Significant difference between groups A and B")
else:
    print("Fail to reject the null hypothesis: No significant difference between groups A and B")
    
#Interpretation:
#If we reject the null hypothesis, it indicates that the new feature (Group B) has a statistically significant impact on conversion rates compared to the old feature (Group A).
#If we fail to reject the null hypothesis, it suggests that there is no significant difference in
#conversion rates between the two features, and further analysis or testing may be needed.
