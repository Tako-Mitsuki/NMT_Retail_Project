# Auto-generated from Copy_of_NMT_Online_Retail.ipynb
# Module: preprocessing

'''
Review this file: the code was heuristically split from the notebook.
Move functions/classes around as needed.
'''

import pandas as pd
import matplotlib.pyplot as plt
# Đọc file CSV với encoding khác (ví dụ: 'ISO-8859-1')
df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1')

# In kích thước và thông tin dữ liệu
print("Kích thước ban đầu:", df.shape)
print(df.info())


df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["InvoiceDate_day"] = df["InvoiceDate"].dt.day

#.dt.day: cho phép truy cập thuộc tính ngày thuộc tính datetime.

df["InvoiceDate_hour"] = df["InvoiceDate"].dt.hour
df["InvoiceDate_month"] = df["InvoiceDate"].dt.month


df["IsCancelled"] = df["Quantity"] < 0
before_rows = len(df)
df = df[df["Quantity"] > 0]   # bỏ các dòng có Quantity âm
removed_cancelled = before_rows - len(df)


missing_before = df["CustomerID"].isna().sum()
df["CustomerID"] = df["CustomerID"].fillna("Guest")
missing_after = df["CustomerID"].isna().sum()


df["Description"] = (
    df["Description"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)


before_rows2 = len(df)
df = df[(df["Quantity"] < 10000) & (df["UnitPrice"] > 0) & (df["UnitPrice"] < 10000)]
removed_outliers = before_rows2 - len(df)

# Quantity >= 10000: Giao dịch không hợp lý bán lẻ. Lỗi nhập dữ liệu
# UnitPrice <=0: đơn bị trả hàng, đơn bị hủy và dự liệu sai. Không đại diện cho hành vi mua hàng nên cần loại bỏ.
# UnitPrice >= 10000: Không thực tế với sản phẩm bán lẻ thông thường.


print("Kích thước cuối:", df.shape)
print(f"Số dòng bị loại bỏ do Quantity âm: {removed_cancelled}")
print(f"Số dòng bị loại bỏ do ngoại lệ (UnitPrice/Quantity): {removed_outliers}")
print(f"Số CustomerID bị thiếu trước khi xử lý: {missing_before}, sau khi xử lý: {missing_after}")
print("Số lượng giao dịch:", df["InvoiceNo"].nunique())
print("Số lượng khách hàng:", df["CustomerID"].nunique())
print("Số lượng sản phẩm:", df["Description"].nunique())


# Tạo cột doanh thu
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Doanh thu theo ngày
revenue_by_day = df.groupby(df["InvoiceDate"].dt.date)["Revenue"].sum()

# Doanh thu theo tháng
revenue_by_month = df.groupby(df["InvoiceDate"].dt.to_period("M"))["Revenue"].sum()

print("🔹 Doanh thu theo ngày (5 dòng đầu):")
print(revenue_by_day.head())
print("\n🔹 Doanh thu theo tháng:")
print(revenue_by_month.head())

# Vẽ
# https://pandas.pydata.org/docs/user_guide/visualization.html#time-series-plotting
revenue_by_day.plot(figsize=(12,6), title="Doanh thu theo ngày")
plt.ylabel("Doanh thu")
plt.show()

revenue_by_month.plot(kind="bar", figsize=(12,6), title="Doanh thu theo tháng")
plt.ylabel("Doanh thu")
plt.show()


top10_quantity = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
top10_revenue = df.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(10)

# grouby("Description"): gom nhóm dữ liệu theo Description và mỗi nhóm = 1 loại sản phẩm.
# ["Quantity"].sum(): tính tổng số lượng bán ra ứng với mỗi sản phẩm.
# ["Revenue"].sum(): tính tổng doanh thu dựa trên mỗi sản phẩm.
# sort_values(ascending=False): sắp xếp thep giá trị giảm dần (ưu tiên các sản phẩm bán nhiều nhất).

print(" Top 10 sản phẩm theo số lượng:")
print(top10_quantity)

print("\n Top 10 sản phẩm theo doanh thu:")
print(top10_revenue)

# Vẽ
top10_quantity.plot(kind="barh", figsize=(10,6), title="Top 10 sản phẩm theo số lượng")
plt.xlabel("Số lượng")
plt.show()

top10_revenue.plot(kind="barh", figsize=(10,6), title="Top 10 sản phẩm theo doanh thu")
plt.xlabel("Doanh thu")
plt.show()


# Doanh thu theo quốc gia
revenue_by_country = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)

# Doanh thu theo giờ trong ngày
revenue_by_hour = df.groupby(df["InvoiceDate"].dt.hour)["Revenue"].sum()

# Doanh thu theo ngày trong tuần (0 = Monday, 6 = Sunday)
revenue_by_weekday = df.groupby(df["InvoiceDate"].dt.dayofweek)["Revenue"].sum()

print(" Doanh thu theo quốc gia (Top 10):")
print(revenue_by_country.head(10))

print("\n Doanh thu theo giờ trong ngày:")
print(revenue_by_hour)

print("\n Doanh thu theo ngày trong tuần (0=Mon, 6=Sun):")
print(revenue_by_weekday)

# Vẽ
revenue_by_country.head(10).plot(kind="bar", figsize=(12,6), title="Doanh thu theo quốc gia (Top 10)")
plt.ylabel("Doanh thu")
plt.show()

revenue_by_hour.plot(kind="bar", figsize=(10,6), title="Doanh thu theo giờ trong ngày")
plt.ylabel("Doanh thu")
plt.show()

revenue_by_weekday.plot(kind="bar", figsize=(10,6), title="Doanh thu theo ngày trong tuần")
plt.ylabel("Doanh thu")
plt.show()


basket_size = df.groupby("InvoiceNo")["Quantity"].sum()
print("Min:", basket_size.min())
print("Max:", basket_size.max())
print("Mean:", basket_size.mean())
print("Median:", basket_size.median())


# max: 15049 sản phẩm trên 1 giỏ hàng. Dữ liệu bị lệch
# median: 151 nhưng max cao dẫn đến bị sai số dữ liệu


from datetime import timedelta

# Ngày tham chiếu (1 ngày sau ngày cuối trong data)
reference_date = df["InvoiceDate"].max() + timedelta(days=1)


# Tính RFM cho từng khách
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",                                   # Frequency
    "Revenue": "sum"                                          # Monetary
}).reset_index()

rfm.rename(columns={"InvoiceDate": "Recency",
                    "InvoiceNo": "Frequency",
                    "Revenue": "Monetary"}, inplace=True)


from sklearn.preprocessing import StandardScaler
# Nếu cột Segment chưa tồn tại, tạo tạm
if "Segment" not in rfm.columns:
    print("⚠️ Cột 'Segment' không tồn tại — tạo mặc định 'ToCluster'")
    rfm["Segment"] = "ToCluster"

# Chỉ lấy khách hàng thuộc nhóm cần phân cụm
# Nếu chưa có cột 'Segment', tạo mặc định
rfm_cluster = rfm.loc[
    rfm.get("Segment", pd.Series(["ToCluster"] * len(rfm))) == "ToCluster",
    ["Recency", "Frequency", "Monetary"]
]


# Chuẩn hóa dữ liệu trước khi phân cụm
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_cluster)

# Trung bình RFM theo từng cụm
if "Cluster" in rfm.columns:
    cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    print("\n📊 Trung bình RFM mỗi cụm:")
    print(cluster_profile)
else:
    print("⚠️ Chưa có cột 'Cluster' — bỏ qua thống kê trung bình cụm.")

# Gán nhãn dễ hiểu cho từng cụm
cluster_labels = {
    0: "Khách giá trị cao",
    1: "Khách trung thành",
    2: "Khách mới",
    3: "Khách rủi ro"
}

if "Cluster" in rfm.columns:
    rfm["Segment"] = rfm["Cluster"].map(cluster_labels).fillna(rfm["Segment"])

print("✅ Hoàn tất gán nhãn phân khúc khách hàng.")



from mlxtend.preprocessing import TransactionEncoder

# Gom sản phẩm theo hóa đơn
basket = df.groupby("InvoiceNo")["Description"].apply(list)

# List of list
transactions = basket.tolist()

# Mã hóa one-hot
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_basket = pd.DataFrame(te_ary, columns=te.columns_)


from datetime import timedelta

# Chia dữ liệu: 80% train, 20% test theo thời gian
split_date = df["InvoiceDate"].quantile(0.8)
df_train = df[df["InvoiceDate"] <= split_date].copy()
df_test  = df[df["InvoiceDate"] > split_date].copy()

# Tạo doanh thu
df_train["Revenue"] = df_train["Quantity"] * df_train["UnitPrice"]
df_test["Revenue"]  = df_test["Quantity"] * df_test["UnitPrice"]

# Reference date cho train
ref_date = df_train["InvoiceDate"].max() + timedelta(days=1)

# RFM từ train
rfm_train = df_train.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (ref_date - x.max()).days,  # Recency
    "InvoiceNo": "nunique",                              # Frequency
    "Revenue": "sum"                                     # Monetary
}).reset_index().rename(columns={"InvoiceDate":"Recency","InvoiceNo":"Frequency","Revenue":"Monetary"})

# Nhãn = chi tiêu ở holdout
future_spend = df_test.groupby("CustomerID")["Revenue"].sum().reset_index().rename(columns={"Revenue":"Future_Monetary"})

# Merge
rfm = rfm_train.merge(future_spend, on="CustomerID", how="left").fillna(0)


rfm["Future_pred"] = model.predict(X)
rfm["CLV_pred_12m"] = rfm["Future_pred"]     # dự đoán cho 12 tháng
rfm["CLV_pred_6m"]  = rfm["Future_pred"]*0.5 # giả định 6 tháng = 1/2 năm


# In ra 5 khách hàng giá trị cao nhất

top5 = rfm.sort_values("CLV_pred_12m", ascending=False).head(5)
print(top5[["CustomerID","Recency","Frequency","Monetary","CLV_pred_6m","CLV_pred_12m"]])


import numpy as np

# Ngưỡng VIP = top 1% Monetary
vip_threshold = rfm["Monetary"].quantile(0.99)
rfm["Segment"] = np.where(rfm["Monetary"] > vip_threshold, "VIP", "ToCluster")


from mlxtend.frequent_patterns import fpgrowth, association_rules

# Frequent itemsets (min_support = 0.01 = 1%)
frequent_itemsets = fpgrowth(df_basket, min_support=0.01, use_colnames=True)

# Sinh luật kết hợp
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Lọc luật mạnh (confidence >= 0.5, lift > 1)
rules_strong = rules[(rules["confidence"] >= 0.5) & (rules["lift"] > 1)]

# Top 10 luật
print(rules_strong.sort_values("lift", ascending=False).head(10))


