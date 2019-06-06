from meta_feature_extraction.metaFeatureExtractionFromDatasets import extractMetaFeaturesFromDatasets
from model_evaluation.modelCreation import createModels
from datasetGenerator import createDatasets
from model_evaluation.modelEvaluation import evaluateModels
from MetaLearner import createMetaLearner

CREATE_DATASETS = False
CREATE_MODELS = True
USE_OLD_MODEL_EVALUATIONS = True

if __name__ == '__main__':
    if CREATE_DATASETS:
        print('Creating Datasets')
        createDatasets()
    extractMetaFeaturesFromDatasets()
    # if CREATE_MODELS:
    #     createModels()
    # evaluateModels(USE_OLD_MODEL_EVALUATIONS)
    createMetaLearner()
