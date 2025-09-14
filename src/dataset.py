import csv
import config

from unicodedata import normalize

class Dataset:
    def __init__(self, train_dataset_path, validation_dataset_path, industry_list_path):
        self.train_dataset = self.init_dataset_by_csvfile(train_dataset_path)
        self.validation_dataset = self.init_dataset_by_csvfile(validation_dataset_path)
        self.industry_list = self.init_industry_list(industry_list_path)
        pass
    
    def init_dataset_by_csvfile(self, path):
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            dataset = {normalize("NFKC", row[1]):normalize("NFKC",  row[0]) for row in reader}
        return dataset
    
    def init_industry_list(self, path):
        industry_list = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            industry_list = {index:normalize("NFKC",  row[0]) for index, row in enumerate(reader)}
        return industry_list
    
if __name__ == "__main__":
    dataset = Dataset(
        config.train_dataset_path,
        config.validation_dataset_path,
        config.industry_list_path
    )
    
    print("Industry list:", dataset.industry_list)
    
    
    

    