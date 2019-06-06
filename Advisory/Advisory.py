from RMI import rmi_build_and_predict
import Path
import numpy as np

def load_data():
    result = {
        'beta1': np.sort(np.loadtxt(Path.BasePath + 'datasets/beta1.csv')),
        'beta2': np.sort(np.loadtxt(Path.BasePath + 'datasets/beta2.csv')),
        'beta24': np.sort(np.loadtxt(Path.BasePath + 'datasets/beta24.csv')),
        'beta25': np.sort(np.loadtxt(Path.BasePath + 'datasets/beta25.csv')),
        'logis1': np.sort(np.loadtxt(Path.BasePath + 'datasets/logis1.csv')),
        'logis2': np.sort(np.loadtxt(Path.BasePath + 'datasets/logis2.csv')),
        'logis45': np.sort(np.loadtxt(Path.BasePath + 'datasets/logis43.csv')),
        'logis44': np.sort(np.loadtxt(Path.BasePath + 'datasets/logis44.csv')),
        'logn1': np.sort(np.loadtxt(Path.BasePath + 'datasets/logn1.csv')),
        'logn2': np.sort(np.loadtxt(Path.BasePath + 'datasets/logn2.csv')),
        'logn51': np.sort(np.loadtxt(Path.BasePath + 'datasets/logn51.csv')),
        'logn52': np.sort(np.loadtxt(Path.BasePath + 'datasets/logn52.csv')),
        'parteo1': np.sort(np.loadtxt(Path.BasePath + 'datasets/parteo1.csv')),
        'parteo2': np.sort(np.loadtxt(Path.BasePath + 'datasets/parteo2.csv')),
        'parto33': np.sort(np.loadtxt(Path.BasePath + 'datasets/parteo33.csv')),
        'parto34': np.sort(np.loadtxt(Path.BasePath + 'datasets/parteo34.csv')),
        'uni1': np.sort(np.loadtxt(Path.BasePath + 'datasets/uni1.csv')),
        'uni2': np.sort(np.loadtxt(Path.BasePath + 'datasets/uni2.csv')),
        'uni19': np.sort(np.loadtxt(Path.BasePath + 'datasets/uni119.csv')),
        'uni20': np.sort(np.loadtxt(Path.BasePath + 'datasets/uni120.csv')),
        'wald1': np.sort(np.loadtxt(Path.BasePath + 'datasets/wald1.csv')),
        'wald2': np.sort(np.loadtxt(Path.BasePath + 'datasets/wald2.csv')),
        'wald63': np.sort(np.loadtxt(Path.BasePath + 'datasets/wald63.csv')),
        'wald64': np.sort(np.loadtxt(Path.BasePath + 'datasets/wald64.csv'))
    }

    return result


if __name__ == '__main__':
    data = load_data()
    for key, value in data.items():
        if len(value < 2000):
            model_pr_layer = [1, 50]
        else:
            model_pr_layer = [1, 1000]
        rmi_build_and_predict(key, value, [1,1000], 0, 0, 0, True)