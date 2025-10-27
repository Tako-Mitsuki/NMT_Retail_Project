# Auto-generated from Copy_of_NMT_Online_Retail.ipynb
# Module: model_training

'''
Review this file: the code was heuristically split from the notebook.
Move functions/classes around as needed.
'''

from sklearn.cluster import KMeans

# Chọn k = 4 (có thể tìm bằng Elbow hoặc Silhouette Score)
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
rfm_cluster["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Gán kết quả vào bảng RFM
rfm.loc[rfm["Segment"] == "ToCluster", "Cluster"] = rfm_cluster["Cluster"]


