from IPython.display import display
from .utils import checking

class Dataset:
    def __init__(self, data, index_cols, target_col='TARGET'): 
        self.data = data
        self.index_cols = index_cols + ['TARGET', '_SAMPLE_']
        
        self.target_col = target_col
        if 'TARGET' not in self.data:
            self.data['TARGET'] = self.data[target_col]
            self.data = self.data.drop(columns=[target_col])

        if '_SAMPLE_' not in self.data.columns:
            self.data['_SAMPLE_'] = self.data['TARGET'].notnull().map({False: 'client_test', True: 'train'})

        # self.category_feats
        # self.numeric_feats
        
    @property
    def feats(self):
        return [col for col in self.data.columns if col not in self.index_cols]
    
    def get_category_feats(self):
        return [col for col in self.data.columns if col in 
                list(set(self.data.select_dtypes(include=['object', 'category']).columns) - set(self.index_cols))]
        
    def get_numeric_feats(self):
        return [col for col in self.data.columns if col in 
                list(set(self.data.select_dtypes(include=['number']).columns) - set(self.index_cols))]
    
    def data_by_sample(self, sample_name):
        return self.data[self.data['_SAMPLE_'] == sample_name].copy()

    def inv_data_by_sample(self, sample_name):
        return self.data[self.data['_SAMPLE_'] != sample_name].copy()

    def y_by_sample(self, sample_name):
        return self.data.loc[self.data['_SAMPLE_'] == sample_name, 'TARGET']

    @property
    def train(self):
        return self.data_by_sample('train')

    @property
    def test(self):
        return self.inv_data_by_sample('train')

    @property
    def client_test(self):
        return self.data_by_sample('client_test')

    @property
    def train_dataset(self):
        return self.update(data=self.train)

    @property
    def y_train(self):
        return self.y_by_sample('train')

    @property
    def index_data(self):
        return self.data[self.index_cols + self.extra_cols]

    @property
    def train_index_data(self):
        return self.train[self.index_cols + self.extra_cols]

    def _to_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_') and not callable(v)}

    # update
    def update(self, **values):
        kwargs = self._to_dict()
        if 'history' in values:
            method, *args = values.pop('history')
            method = method.__name__
            values['history'] = kwargs['history'] + [(method, *args)]

        kwargs.update(values)
        return self.__class__(**kwargs)

    def show_sample(self, sample):
        display(self.data_by_sample(sample).head(4))
        print('-'*88)
        display(checking(self.data_by_sample(sample)[self.feats]))
    
    def drop(self, cols):
        return self.update(data=self.data.drop(columns=cols))

    def add_column(self, col_name, values):
        data = self.data.copy()
        data[col_name] = values
        return self.update(data=data)
