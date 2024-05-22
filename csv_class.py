from dataclasses import dataclass

@dataclass
class Result:
    timestamp: str
    backbone: str
    aggregation: str
    checkpoint_path: str
    pca: bool
    pca_dim: int
    dataset: str
    resize_H: int
    resize_W: int
    corruption: str
    severity: int
    recall_1: float
    recall_5: float
    recall_10: float
    recall_20: float
    
@dataclass
class CorruptedResult:
    timestamp: str
    backbone: str
    aggregation: str
    checkpoint_path: str
    pca: bool
    pca_dim: int
    dataset: str
    result_H: int
    result_W: int
    corruption: str
    severity: int
    mCR_1: float
    mCR_5: float
    mCR_10: float
    mCR_20: float
    mCAP_1: float
    mCAP_5: float
    mCAP_10: float
    mCAP_20: float