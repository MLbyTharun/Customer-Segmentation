📊 Customer Segmentation using K-Means Clustering
Machine Learning Project | Unsupervised Learning | Customer Insights

This project performs Customer Segmentation using the popular Mall Customer Dataset, helping businesses understand different customer groups based on:

Age

Annual Income

Spending Score

The goal is to provide actionable insights that can help companies improve marketing strategies, personalized targeting, and customer retention.

🚀 Why This Project Matters (Client-Oriented)

Businesses often spend money inefficiently because all customers are treated the same.
This model solves that problem by:

✔ Identifying high-value customers
✔ Separating budget shoppers from premium spenders
✔ Helping businesses design targeted marketing campaigns
✔ Improving product recommendations
✔ Reducing unnecessary marketing costs

If you are a business, this segmentation helps you increase revenue with data-driven decisions.

🧠 What the Model Does

The project uses K-Means Clustering, an unsupervised learning algorithm, to discover natural customer groups.

Key Features:

Data preprocessing & gender encoding

Clustering based on income, spending score & age

Automatic cluster naming (example: Premium High Spenders)

Visualizations:

Elbow Method

Silhouette Analysis

3D Cluster Plot

Heatmaps

🔧 Tech Stack

Python

Pandas

Scikit-Learn

NumPy

Seaborn / Matplotlib

🔍 How the Segmentation Works
1️⃣ Data Loading & Cleaning

Gender encoding + null checks.

2️⃣ Feature Selection

Uses:

Age

Annual Income (k$)

Spending Score (1-100)

3️⃣ Find Optimal Number of Clusters

Based on:

Elbow Method

Silhouette Score

4️⃣ Apply K-Means

Generates customer groups.

5️⃣ Assign Human-Readable Cluster Names

Example:

Cluster	Meaning
Premium High Spenders	Young/old customers with high income & high spending
Rich Low Spenders	High income but low interest
Budget High Spenders	Low income but big spenders
Low Income Low Spenders	Conservative buyers
Mid-Tier Average	Balanced segment
6️⃣ Visualize Insights

High-quality plots help understand each cluster clearly.

📈 Business Insights (Example)

After running the model, you may discover insights like:

Cluster 1: High-income customers who spend aggressively (ideal for premium product marketing)

Cluster 3: Low-income customers with low engagement (focus on discounts or retention strategies)

Cluster 4: Young customers with high spending (target for modern/e-commerce campaigns)

These insights help companies save money and increase ROI by targeting the right audience.

▶️ How to Run the Project
pip install -r requirements.txt
python segmentation.py


Or open the notebook:

jupyter notebook segmentation.ipynb

📬 Want This for Your Business?

If you're looking for:

Custom customer segmentation

Marketing insights

Predictive analytics

Dashboard development

I can build a complete ML system tailored for your business.

📩 Contact me on Fiverr: MLbyTharun

⭐ If This Helped You

Please ⭐ star the repository — it motivates further projects like this!
