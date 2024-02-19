from dataclasses import dataclass

@dataclass
class Config:
    TARGET_FEATURE = 'price_range'
    model_path = 'artifacts/model.pkl'
    preprocessor_path  = 'artifacts/preprocessor_transform.pkl'