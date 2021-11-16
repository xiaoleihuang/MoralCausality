from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def paint(embed,label):
    pca = PCA(n_components=11)
    reduced_data_pca = pca.fit_transform(embed)
    label_list = list(set(label))
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray','pink']

    for i in range(len(colors)):
        x = reduced_data_pca[:, 0][label_list.index(label) == i]
        y = reduced_data_pca[:, 1][label == i]
        plt.scatter(x, y, c=colors[i])
        plt.show()
        print('a')
