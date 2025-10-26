#Monthly Sales Trend

import matplotlib.pyplot as plt

# Sample data: 12 months of website traffic (in thousands)
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
traffic = [150, 165, 180, 210, 200, 220, 240, 230, 215, 250, 265, 280]

# Create a line plot
plt.figure(figsize=(10, 6))
plt.plot(months, traffic, marker='o', linestyle='--', color='b')

# Add titles and labels
plt.title("Monthly Website Traffic Trend")
plt.xlabel("Month")
plt.ylabel("Traffic (in Thousands)")
plt.grid(True)

# Show the plot
plt.show()