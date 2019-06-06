import rpy2.robjects as robj
import Path

def extractMetaFeaturesFromDatasets():
    print("Extracing meta-features")
    robj.globalenv['path'] = Path.BasePath + "datasets/"
    robj.r.source(Path.BasePath + "Acquisition/meta_feature_extraction/MetaFeatures.r")
    print("Meta-features extracted")

