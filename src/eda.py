# Auto-generated from Copy_of_NMT_Online_Retail.ipynb
# Module: eda

'''
Review this file: the code was heuristically split from the notebook.
Move functions/classes around as needed.
'''



# Tính median
median_val = basket_size.median()
print("Median basket size:", median_val)

# Lọc theo median (ví dụ < 5 lần median)
threshold = 5 * median_val
basket_size_filtered = basket_size[basket_size < threshold]

print("Số giỏ ban đầu:", len(basket_size))
print("Số giỏ sau khi lọc:", len(basket_size_filtered))

# Vẽ lại histogram
plt.figure(figsize=(10,6))
basket_size_filtered.plot(kind="hist", bins=50, title="Phân phối kích cỡ giỏ hàng (lọc theo median)")
plt.xlabel("Số sản phẩm mỗi giỏ")
plt.show()


import matplotlib.pyplot as plt

plt.scatter(rules["support"], rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Scatter plot luật kết hợp")
plt.show()


import networkx as nx

G = nx.from_pandas_edgelist(
    rules_strong.head(20), "antecedents", "consequents", edge_attr=True
)

plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, with_labels=True, node_size=1500, node_color="lightblue", font_size=10)
plt.title("Mạng lưới(graph mạng) luật kết hợp (Top 20)")
plt.show()


