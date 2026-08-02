import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Sample transaction data
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Bread', 'Butter'],
    ['Milk', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

# Convert transactions into DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

# Generate Association Rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print("\n===== Frequent Itemsets =====")
print(frequent_itemsets)

print("\n===== Association Rules =====")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

print("\n===== Product Placement Suggestions =====")
for _, row in rules.iterrows():
    print(f"Place {list(row['antecedents'])} near {list(row['consequents'])}")

# Plot Support Graph
plt.figure(figsize=(8,4))
plt.bar(
    frequent_itemsets['itemsets'].astype(str),
    frequent_itemsets['support']
)
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.title("Frequent Itemsets")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
