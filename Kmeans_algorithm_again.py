import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Loaded the scraped dataset into a pandas DataFrame
file_path = "F:\\Masters_documents\\Module2_and_3_Python\\Testing\\Final_Dataset_LinkedIn.xlsx"
df = pd.read_excel(file_path)
df["SKILLS"]=df["SKILLS"].str.upper()

data_engineer_skills=["PYTHON","MYSQL","BIG DATA","POWER BI","PYSPARK","ETL"]

# Created binary columns for each skill
for skill in data_engineer_skills:
    df[skill] = df['SKILLS'].apply(lambda x: 1 if skill in x else 0)

dff = df
data = df[data_engineer_skills]

#Kmeans algorithm and found the optimal number of clusters through elbow curve method 
cluster_centers=[]
k_values=[]

for k in range(1,20):
    kmeans = KMeans(n_clusters=k,n_init=10,init="k-means++")
    kmeans.fit(data)
    
    cluster_centers.append(kmeans.inertia_)
    k_values.append(k)

#By observing the plot, concluded the data requires only 3 clusters  

figure=plt.subplots(figsize=(12,6))
plt.plot(k_values,cluster_centers,'o-',color='orange')
plt.xlabel("Number of clusters")
plt.ylabel("cluster inertia")
plt.show()

for i in range(0,4):
    data = df[data_engineer_skills]
    # defining the kmeans function with initialization as k-means++
    kmeans = KMeans(n_clusters=3,n_init=10,init="k-means++")

    # fitting the k means algorithm on scaled data
    kmeans.fit(data)

    df['Cluster'] = kmeans.labels_
    cluster_labels= kmeans.labels_


    silhouette_avg = silhouette_score(data, cluster_labels)
    print("Silhouette Score:", silhouette_avg)

    # Create a list of cluster labels
    cluster_names = df['Cluster'].unique()
    print(cluster_names)

    threshold_value = 0.00

    # Create a list of cluster names where values are greater than the threshold
    selected_clusters = []
    selected_cluster_data = pd.DataFrame()  

    skills_to_check = ["PYTHON", "BIG DATA", "MYSQL","PYSPARK"] #skills of highest priority
    check=["POWER BI","ETL"] #skills with little priority

    # Check each cluster
    for cluster_name in cluster_names:
        cluster_data = df[df['Cluster'] == cluster_name]
        # x=cluster_data.iloc[:,:12]
        # x.to_csv(f'F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_f_{cluster_name}.csv', index=False)
        cluster_mean = cluster_data[data_engineer_skills].mean()
        check_skills = cluster_data[skills_to_check].mean()
        check_double_skills= cluster_data[check].mean()
        print(f'Cluster_{cluster_name}')
        print(cluster_mean)

        if not any(value == threshold_value for value in check_skills): #and (any(value>=threshold_value for value in check_double_skills)):
            selected_clusters.append(f"Cluster_{cluster_name}")
            selected_cluster_data = pd.concat([selected_cluster_data, cluster_data], ignore_index=True)

    df = selected_cluster_data


    # Print the list of selected cluster names
    print("Selected clusters:", selected_clusters)

df.to_csv("F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_final4_6.csv", index=False)












# threshold_value=0.15

# # Create a list of dictionary names where values are greater than the threshold
# selected_clusters = []

# # Check each dictionary
# for cluster_name, cluster_dict in [("Cluster_0", Cluster_0_mean), ("Cluster_1", Cluster_1_mean), ("Cluster_2", Cluster_2_mean)]:
#     if all(value > threshold_value for value in cluster_dict.values()):
#         selected_clusters.append(cluster_name)

# # Print the list of selected cluster names
# print("Selected clusters:", selected_clusters)






# Cluster_0 = df[df['Cluster'] == 0]
# Cluster_0=Cluster_0.iloc[:,:12]
# Cluster_0_mean= Cluster_0[data_engineer_skills].mean()
# print("cluster0",Cluster_0_mean)

# Cluster_1= df[df['Cluster']==1]
# Cluster_1=Cluster_1.iloc[:,:12]
# Cluster_1_mean=Cluster_1[data_engineer_skills].mean()
# print("cluster1",Cluster_1_mean)

# Cluster_2=df[df['Cluster']==2]
# Cluster_2=Cluster_2.iloc[:,:12]
# Cluster_2_mean=Cluster_2[data_engineer_skills].mean()
# print("cluster2", Cluster_2_mean)

# print(Cluster_0.head())
# print(Cluster_1.head())
# print(Cluster_2.head())

# # Save the DataFrame to a CSV file
# Cluster_0.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output0_0.csv', index=False)
# Cluster_1.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output0_1.csv',index=False)
# Cluster_2.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output0_2.csv',index=False)










# # Create a K-Means model with the specified number of clusters
# # kmeans = KMeans(n_clusters=7, random_state=42,n_init=10)
# kmeans = KMeans(n_clusters=3,n_init=10,init="k-means++")

# # Fit the model to your data
# kmeans.fit(data)

# # Get the cluster assignments for each data point
# cluster_labels = kmeans.labels_

# df['Cluster'] = cluster_labels

# # Get the cluster centroids
# centroids = kmeans.cluster_centers_

# # Print the centroids for each cluster
# for i, centroid in enumerate(centroids):
#     print(f"Centroid for Cluster {i + 1}: {centroid}")



# Cluster_0 = df[df['Cluster'] == 0]
# Cluster_0=Cluster_0.iloc[:,:12]
# Cluster_1= df[df['Cluster']==1]
# Cluster_1=Cluster_1.iloc[:,:12]
# Cluster_2=df[df['Cluster']==2]
# Cluster_2=Cluster_2.iloc[:,:12]

# # # data = [Cluster_0,Cluster_1]


# # Cluster_3 = df[df['Cluster'] == 3]
# # Cluster_3=Cluster_0.iloc[:,:12]
# # Cluster_4= df[df['Cluster']==4]
# # Cluster_4=Cluster_1.iloc[:,:12]

# # Cluster_5 = df[df['Cluster'] == 5]
# # Cluster_5=Cluster_0.iloc[:,:12]
# # Cluster_6= df[df['Cluster']==6]
# # Cluster_6=Cluster_1.iloc[:,:12]

# # Cluster_7= df[df['Cluster']==6]
# # Cluster_7=Cluster_7.iloc[:,:12]
# # Cluster_8= df[df['Cluster']==6]
# # Cluster_8=Cluster_8.iloc[:,:12]

# # silhouette_avg = silhouette_score(data, cluster_labels)
# # print("Silhouette Score:", silhouette_avg)

# # print(Cluster_0.head())
# # print(Cluster_1.head())
# # print(Cluster_2.head())

# # print(Cluster_3.head())
# # print(Cluster_4.head())

# # Save the DataFrame to a CSV file
# Cluster_0.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_0.csv', index=False)
# Cluster_1.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_1.csv',index=False)
# Cluster_2.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_2.csv',index=False)

# # Cluster_3.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_3.csv',index=False)
# # Cluster_4.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_4.csv',index=False)

# # Cluster_5.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_5.csv', index=False)
# # Cluster_6.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output_6.csv',index=False)

# # Cluster_7.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output.7.csv',index=False)
# # Cluster_8.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output.8.csv',index=False)
# # # Cluster_4.to_csv('F:\\Masters_documents\\Module2_and_3_Python\\Testing\\output.4.csv',index=False)













# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#print(np.unique(cluster_labels))

# # Get the coordinates of the cluster centers
# centers = kmeans.cluster_centers_


# # Create three separate DataFrames based on the cluster labels
# cluster_0_df = data[cluster_labels == 0]
# cluster_1_df = data[cluster_labels == 1]
# cluster_2_df = data[cluster_labels == 2]

# # Optionally, you can convert these DataFrames to lists if needed
# cluster_0_list = cluster_0_df.values.tolist()
# cluster_1_list = cluster_1_df.values.tolist()
# cluster_2_list = cluster_2_df.values.tolist()

# # Display the first few rows of each cluster (optional)
# print("Cluster 0:")
# print(cluster_0_df.head())
# print("\nCluster 1:")
# print(cluster_1_df.head())
# print("\nCluster 2:")
# print(cluster_2_df.head())
