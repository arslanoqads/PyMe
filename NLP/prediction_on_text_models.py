# Initialize models
vectorizer = CountVectorizer(min_df=1)
pca = RandomizedPCA(n_components=50, whiten=True)
km = KMeans(n_clusters=2, init='random', n_init=1, verbose=1)

# Fit models
X = vectorizer.fit_transform(sentences)
X2 = pca.fit_transform(X)
km.fit(X2)

# Predict with models
X_new = vectorizer.transform(["hello world"])
X2_new = pca.transform(X_new)
km.predict(X2_new)