import numpy as np

labels = np.random.randint(0, 2, (3, 4))
probabilities = np.random.randint(2, 10, (3, 4))


print(labels)
print(probabilities)

# labels = np.array(labels_list)
# probabilities = np.array(probabilities)
inverse_labels = np.array(~np.array(labels, dtype=bool), dtype=int)
pos = probabilities*labels
neg = probabilities*inverse_labels

print(pos)
print(neg)

labels = np.count_nonzero(labels, axis=0)
labels = np.array(labels>2.5, dtype=int) # Majority voting
print(labels)

inverse_labels = np.array(~np.array(labels, dtype=bool), dtype=int)

probabilities = np.array(pos*labels + neg*inverse_labels, dtype=float)

probabilities[probabilities==0] = np.nan
print(probabilities)
probabilities = np.nanmean(probabilities, axis=0)
print(probabilities)

# pos[:, labels==0] = np.nan
# neg[:, labels==1] = np.nan

