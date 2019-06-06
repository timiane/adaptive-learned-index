# import keras
import pandas
import numpy as np
import matplotlib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

features = ["CC", "LR", "SNNR", "Inst", "var", "DD", "kurt", "cor"]

features_results = ["CC", "LR", "SNNR", "Inst", "var", "DD", "kurt", "cor", "size", "inference", "accuracy"]


split = 8
def principal_component_analysis(dataset):
    X = dataset[:, 0:split].astype(float)
    Y = dataset[:, split]
    YDf = pandas.DataFrame({'target':Y})

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pandas.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pandas.concat([principalDf, YDf], axis=1)

    colors = ['red', 'green', 'blue', 'yellow', 'cyan', '#a8a8a8']
    targets = ['beta', 'logis', 'logn', 'uni', 'wald', 'parteo']


    for target, color in zip(targets, colors):
        indiciesToKeep = finalDf['target'] == target
        plt.scatter( finalDf.loc[indiciesToKeep, 'principal component 1'], finalDf.loc[indiciesToKeep, 'principal component 2'], c=color, s=50)
        plt.legend(targets)
        plt.xlabel('PC1 - 0.415562')
        plt.ylabel('PC2 - 0.336085')
        plt.grid()

    coeff = np.transpose(pca.components_[0:2, :])
    for i in range(coeff.shape[0]):
        plt.arrow(0, 0, coeff[i, 0] * 4.5, coeff[i, 1] * 4.5, color='black', alpha=0.8)
        plt.text(coeff[i, 0] * 4.65, coeff[i, 1] * 4.65, features[i], color='black', ha='center', va='center')

    plt.show()
    print(pca.explained_variance_ratio_)



def principal_component_analysis_for_results():
    dataframe033033033 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe05025025 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe02505025 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe02502505 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe080101 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe010801 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    dataframe010108 = pandas.read_csv('metaFeaturesCC.csv', header=None)
    # dataframe001 = pandas.read_csv('metaFeaturesNames.csv', header=None)
    results1 = pandas.read_csv('results_033_033_033.csv')
    results2 = pandas.read_csv('results_05_025_025.csv')
    results3 = pandas.read_csv('results_025_05_025.csv')
    results4 = pandas.read_csv('results_025_025_05.csv')
    results5 = pandas.read_csv('results_08_01_01.csv')
    results6 = pandas.read_csv('results_01_08_01.csv')
    results7 = pandas.read_csv('results_01_01_08.csv')
    # results5 = pandas.read_csv('results_0_0_1.csv')

    results1 = results1.sort_values(by=['name'])
    results2 = results2.sort_values(by=['name'])
    results3 = results3.sort_values(by=['name'])
    results4 = results4.sort_values(by=['name'])
    results5 = results5.sort_values(by=['name'])
    results6 = results6.sort_values(by=['name'])
    results7 = results7.sort_values(by=['name'])
    # results5 = results5.sort_values(by=['name'])

    dataframe033033033 = dataframe033033033.sort_values(by=[7])
    dataframe05025025 = dataframe05025025.sort_values(by=[7])
    dataframe02505025 = dataframe02505025.sort_values(by=[7])
    dataframe02502505 = dataframe02502505.sort_values(by=[7])
    dataframe080101 = dataframe080101.sort_values(by=[7])
    dataframe010801 = dataframe010801.sort_values(by=[7])
    dataframe010108 = dataframe010108.sort_values(by=[7])
    # dataframe001 = dataframe001.sort_values(by=[9])

    dataframe033033033['size'] = 0.33
    dataframe033033033['inference'] = 0.33
    dataframe033033033['accuracy'] = 0.33

    dataframe05025025['size'] = 0.5
    dataframe05025025['inference'] = 0.25
    dataframe05025025['accuracy'] = 0.25

    dataframe02505025['size'] = 0.25
    dataframe02505025['inference'] = 0.5
    dataframe02505025['accuracy'] = 0.25

    dataframe02502505['size'] = 0.25
    dataframe02502505['inference'] = 0.25
    dataframe02502505['accuracy'] = 0.5

    dataframe080101['size'] = 0.8
    dataframe080101['inference'] = 0.1
    dataframe080101['accuracy'] = 0.1

    dataframe010801['size'] = 0.1
    dataframe010801['inference'] = 0.8
    dataframe010801['accuracy'] = 0.1

    dataframe010108['size'] = 0.1
    dataframe010108['inference'] = 0.1
    dataframe010108['accuracy'] = 0.8

    # dataframe001['size'] = 0
    # dataframe001['inference'] = 0
    # dataframe001['accuracy'] = 1


    dataframe = dataframe033033033.append([dataframe05025025, dataframe02505025, dataframe02502505, dataframe080101, dataframe010801, dataframe010108], ignore_index=True)
    dataframe = dataframe[[0, 1, 2, 3, 4, 5, 6, 7, 'size', 'inference',  'accuracy']]
    print(dataframe.dtypes)

    cols_to_norm = [0, 1, 2, 3, 4, 5, 6, 7, 'size', 'inference', 'accuracy']
    dataframe[cols_to_norm] = dataframe[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    results = results1.append([results2, results3, results4, results5, results6, results7], ignore_index=True)

    results['best'] = results[['nnr', 'nn', 'ir', 'lr', 'mvr', 'spline']].idxmax(axis=1)
    results = results.values
    metafeatures = dataframe.values

    X = metafeatures[:, 0:11].astype(float)
    Y = results[:, 7]
    YDf = pandas.DataFrame({'target':Y})

    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pandas.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pandas.concat([principalDf, YDf], axis=1)

    colors = ['red', 'green', 'blue']
    targets = ['lr', 'spline', 'ir']


    for target, color in zip(targets, colors):
        indiciesToKeep = finalDf['target'] == target
        plt.scatter( finalDf.loc[indiciesToKeep, 'principal component 1'], finalDf.loc[indiciesToKeep, 'principal component 2'], c=color, s=50)
        plt.legend(targets)
        plt.xlabel('PC1 - 0.415562')
        plt.ylabel('PC2 - 0.336085')
        plt.grid()

    coeff = np.transpose(pca.components_[0:2, :])
    for i in range(coeff.shape[0]):
        plt.arrow(0, 0, coeff[i, 0] * 4.5, coeff[i, 1] * 4.5, color='black', alpha=0.8)
        plt.text(coeff[i, 0] * 4.65, coeff[i, 1] * 4.65, features_results[i], color='black', ha='center', va='center')

    plt.savefig('pca_for_results.png')
    print(pca.explained_variance_ratio_)


def plotData():
    import matplotlib.pyplot as plot
    logar = np.loadtxt("data/zipf1.csv")
    ran = np.array(range(0, len(logar)))
    plot.xlabel("x", fontweight='bold')
    plot.ylabel("y", fontweight='bold')
    plot.plot(logar, ran,)
    plot.show()


if __name__ == '__main__':
    # plotData()
    dataframe = pandas.read_csv('metaFeaturesPCA.csv', header=None)
    dataset = dataframe.values
    principal_component_analysis(dataset)
    #principal_component_analysis_for_results()
    #NN = metaLearnerNN()
    #loss, model = train(NN, dataset)
    #print(loss)


