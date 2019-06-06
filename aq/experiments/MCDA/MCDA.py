import math

def weighted_sum_method(benificial, non_benificial, weights):

    for column in benificial:
        benificial[column] = benificial[column]/benificial[column].max()

    for column in non_benificial:
        if(column == 'size'):
            non_benificial[column] = non_benificial[column].map(math.log2)
        non_benificial[column] = non_benificial[column].min() / non_benificial[column]

    normalized_matrix = non_benificial.join(benificial)
    return (normalized_matrix.iloc[:,0] * weights[0]) + (normalized_matrix.iloc[:,1] * weights[1]) + (normalized_matrix.iloc[:,2] * weights[2])

