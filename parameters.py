from sentiment_score import confidence as confidence_prev
from sentiment_score import alpha as alpha_prev
from sentiment_score import exponent as exponent_prev
from sentiment_score import volume as volume_prev

# Values taken from sentiment_resluts_NIFTY_50_INDEXSE.csv
prev_close_value=[22123.65]
day_range=[22163.60 , 22516.00]
year_range=[17204.65 , 22526.60]

# Calculate Confidence 
day_range_pct_change = (day_range[1]-day_range[0])/day_range[0] 
confidence = (1 - (day_range_pct_change))/2 
confidence=(confidence+confidence_prev)/2.5

 # Calculate Alpha (α)
alpha_pct_change = (day_range[0] - prev_close_value[0]) / prev_close_value[0]  
alpha = (1 - abs(alpha_pct_change))/2 
alpha=(alpha+alpha_prev)/2.5

# Calculate Volume
volume = (day_range[1]-day_range[0])/10
volume=(volume+volume_prev)/2.5

# Calculate Exponent 
year_range_difference = year_range[1] - year_range[0]  
exponent = (year_range_difference / (year_range[1]))*10  
exponent=(exponent+exponent_prev)/4

print("Confidence:", confidence)
print("Alpha:", alpha)
print("Volume:", volume)
print("Exponent:", exponent)

