
# coding: utf-8

# In[1]:


import numpy as np
import numpy.lib.recfunctions as rfn # for record arrays
import pandas as pd
from scipy.stats import linregress
import os
import time
import inspect 
from datetime import datetime
from functools import reduce
import multiprocessing as mp
from tqdm import tqdm # for progress bars on iterables

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")


# In[2]:


class ProcessData:
    """Class for processing the HomeCredit datasets.
        
    Data is converted from CSV to HDF5 for faster future loading of raw data and saved to Numpy record arrays after
    processing for a significant speed gain in future loading if no further processing is needed.
    """
    def __init__(self, filename, save_to_numpy=True):
        self._input_path = '../input/'
        self.__numpy_path = '../input/numpy'
        self._id_var = 'SK_ID_CURR' 
        self.__filename = filename
        self.__npy_filename_train = filename.replace('.csv', '_train.npy')
        self.__npy_filename_test = filename.replace('.csv', '_test.npy')
        self.__save_to_numpy = save_to_numpy
        self.__ids_train = pd.read_csv(os.path.join(self._input_path, 'application_train.csv'),
                                       usecols=[self._id_var, 'TARGET'])
        self.__ids_test = pd.read_csv(os.path.join(self._input_path, 'sample_submission.csv'), usecols=[self._id_var])
        self.__ids = self.__ids_train.append(self.__ids_test, sort=False, ignore_index=True)
        
    def __show_time_message(self):
        """Shows the time elapsed
        
        self.__init_time first has to be set, then calling this method shows elapsed time and resets self.__init_time.
        """
        if self.__init_time != None:
            message = 'Loaded and processed ' + self.__filename.replace('.csv', '') + ' data. '
            print('[+] '+message+' {:.1f}s'.format((time.time() - self.__init_time)))
            self.__init_time = None
      
    @staticmethod
    def dataframe_objects_to_categories(df):
        """Convert object data types to category data types for data compression (i.e. memory reduction at runtime)
        
        Args:
            df: dataframe to convert.
            
        Returns:
            Modified data frame.
        """
        for col in df.columns:
            if df[col].dtype=='object':
                df[col] = df[col].astype('category')
        return df
    
    @staticmethod
    def get_csv_or_hd5(filename):
        """Load CSV or HDF5 files.
        
        If the *.hd5 file of the specified CSV file exists, it loads it. Otherwise it loads the CSV
        and creates the HDF5 for future loading.
        
        Args:
            filename: filename to load (with *.csv extension).
            
        Returns:
            Pandas dataframe with the data from the specified file.
        """
        filename_h5 = filename.replace('.csv', '.h5')
        if os.path.isfile(filename_h5):
            df = pd.read_hdf(filename_h5)
        else:
            df = pd.read_csv(filename)
            df.to_hdf(filename_h5, key='hc', format='fixed')
        return df
            
    @staticmethod
    def _trendline(data, y=None):
        """Compute the constant coefficient of a time series, interpreted as the trend.
        
        If used with the HomeCredit variable MONTHS_BALANCE, 
        
        Args:
            data: Pandas Series to use as the independent variable in the simple linear regression.
            y: Pandas Series to use as the dependent variable in the simple linear regression. If None,
            a range [0, n] is generated automatically.
            
        Returns:
            Trendline slope for the data given as input.
        """
        X = data.values # convert to Numpy array
        len_x = len(X)
        if len_x == 1: # if there is only one element, do not compute the slope
            return np.nan
        if not isinstance(y, pd.Series):
            y = np.arange(len(X))
        slope = linregress(y,X)[0]
        return slope

                
    def load_data(self, load_from_numpy=False):
        """Loads a CSV dataset, applies feature engineering and exports to numpy for future use

        Args:
            load_from_numpy: boolean to load from numpy. If True, the numpy file is loaded, saving a significant
            amount of time. 
            
        Returns:
            List containing both train and test data (see __process_data) 
        """
        self.__init_time = time.time()
        if load_from_numpy:
            try:
                X_train = np.load(os.path.join(self.__numpy_path, self.__npy_filename_train))
                X_test = np.load(os.path.join(self.__numpy_path, self.__npy_filename_test))
                self.__show_time_message()
                return X_train, X_test
            except:
                print('[-] Could not load numpy files. Restarting data processing from scratch...')
                self.load_data(load_from_numpy=False)
        else:
            df = self.get_csv_or_hd5(os.path.join(self._input_path, self.__filename))
                
            if '_train' in self.__filename: # we need to append the application test data to the train set
                test_filename = self.__filename.replace('_train', '_test')
                test_df = self.get_csv_or_hd5(os.path.join(self._input_path, test_filename))
                df = df.append(test_df, sort=False, ignore_index=True)
            X = self.__process_data(df)
            self.__show_time_message()
            return X
        
    def __export_to_numpy(self, X_train, X_test):
        """Saves the dataframes to numpy files
        
        Numpy files are much faster to load and can also save time in the future by removing the need of preprocessing
        on every run even if no changes were made.

        Args:
            X_train: numpy array holding the train dataset
            X_test: numpy array holding the test dataset
        """
        if self.__save_to_numpy:
            try:
                np.save(os.path.join(self.__numpy_path, self.__npy_filename_train), X_train)
                np.save(os.path.join(self.__numpy_path, self.__npy_filename_test), X_test)
            except:
                print('[!] Could not export data to numpy files, continuing...')
                
    def __clear_large_days(self, df):
        """Transform very large days to NAs

        As per Martin Kotek, competition host:
        "Value 365243 denotes infinity in DAYS variables in the datasets, therefore you can consider them NA values."
        https://www.kaggle.com/c/home-credit-default-risk/discussion/57247

        Args:
            df: pandas dataframe containing columns to be transformed.

        Returns:
            The modified dataframe.
        """
        cols_to_replace = [col for col in df.columns if 'DAYS' in col]
        if len(cols_to_replace) > 0:
            df[cols_to_replace] = df[cols_to_replace].replace(365243, np.nan)
        return df
        
    def __transform_days_to_years(self, df):
        """Transform DAYS variables into positive years by (1) dividing by 365 and (2) switching the integer's sign.

        Args:
            df: pandas dataframe containing columns to be transformed.

        Returns:
            The modified dataframe.
        """
        cols_to_replace = [col for col in df.columns if 'DAYS' in col]
        if len(cols_to_replace) > 0:
            df[cols_to_replace] = (df[cols_to_replace] / -365) # switch sign and divide by 365 (tough luck leap years)
            replaced_colnames = [col.replace('DAYS', 'YEARS') for col in cols_to_replace] # DAYS becomes NEW_YEARS
            df.rename(columns=dict(zip(cols_to_replace, replaced_colnames)), inplace=True) # rename the relevant columns
        return df
    
    @staticmethod
    def _encode_data(df, encoding_type='label', nan_as_category=True, verbose=False, categorical_columns=[]):
        """Apply encoding to a dataframe (either label encoding or one hot encoding)
        
        Args:
            df: pandas dataframe containing columns to be encoded.
            encoding_type: string with either 'label' or 'ohe'.
            nan_as_category: boolean argument to indicate whether NaNs should be treated as a category or not.
            verbose: boolean to print output information or not
            categorical_columns: list of column names to be encoded, if empty, all object type columns are encoded.
            
        Returns:
            The modified dataframe.
            
        Raises:
            ValueError: if the encoding type is not among the available types.
        """
        if len(categorical_columns) == 0: # if no column names were specified, process all object-type columns
            categorical_columns = [col for col in df.columns if df[col].dtypes == 'object']
        if len(categorical_columns) == 0: # if no object-type columns are available, let's skip encoding
            if verbose:
                print('[!] No columns to encode.')
            return df
        if verbose:
            print('[!] Data shape before '+encoding_type+': {}.'.format(df.shape))
            
        if encoding_type == 'label':
            df[categorical_columns] = df[categorical_columns].apply(lambda x: pd.factorize(x)[0] + 1)
        elif encoding_type == 'ohe':
            df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
        else:
            raise ValueError('Encoding can only be label or ohe.')
            
        if verbose:
            print('[!] Data shape after '+encoding_type+': {}.'.format(df.shape))
        return df
    
    @staticmethod
    def _count_categories_pivot(df, category='', name='', by_id='SK_ID_CURR'):
        """Counts the number of categories per ID and creates one column with the count per ID for the said category.
        
        For example, ID 001 has variable (category) animal and there are 5 rows (3 cats and 2 dogs), then only one row
        will exist for ID 001 (instead of 5) as well as two new columns: cat_count and dog_count, with values of 3 and 2
        respectively.
        
         Args:
            df: pandas dataframe containing columns to be encoded.
            category: variable name of the variable we wish to count categories for.
            name: name to be inserted into the new variable names created for the count (used as identifier for origin\
            dataset)
            
        Returns:
            The modified dataframe.
        """
        df[category] = df[category].replace('XNA', np.nan)
        df2 = df.groupby([by_id, category]).agg({category: 'count'})
        df2.rename(columns={category: category + '_COUNT'}, inplace=True)
        df2 = df2.reset_index()
        df3 = df2.pivot(index=by_id, columns=category, values=category + '_COUNT').fillna(0)
        df3 = df3.add_prefix('N_'+name+'_'+category+'_').add_suffix('_COUNT')
        df3.columns = map(str.upper, df3.columns)
        df3.columns = df3.columns.str.replace(' ', '_')
        return df3
    
    def __clean_data_general(self, df):
        """Applies the general data cleaning methods and calls the method for dataset-specific cleaning.

        Args:
            df: pandas dataframe to be cleaned.

        Returns:
            The modified dataframe.
        """
        if 'TARGET' in df.columns:
            df.drop(['TARGET'], axis=1, inplace=True)        
        df = self.__clear_large_days(df)
        df = self.__transform_days_to_years(df)
        df = self._clean_data(df) # specific data cleaning
        return df
    
    def _clean_data(self, df):
        """Cleans data.
        
        The method is meant to be modified via inheritance to add cleaning specific to the dataset on which it will be
        used on.

        Args:
            df: pandas dataframe to be cleaned.

        Returns:
            Pandas dataframe containing the modified data.
        """
        return df
    
    def _feature_engineering(self, df):
        """Adds new features.
        
        The method is meant to be modified via inheritance to add cleaning specific to the dataset on which it will be
        used on.

        Args:
            df: pandas dataframe to add features to.

        Returns:
            Pandas dataframe containing the modified data.
        """
        return df
    
    def __process_data(self, df):
        """Processes the dataset by applying data cleaning, feature engineering and transformation into numpy record
        arrays.

        Args:
            df: pandas dataframe to be processed.

        Returns:
            List containing two numpy record arrays, X_train for the train data and X_test for the test data.
        """
        df = self.__clean_data_general(df)
        df = self._feature_engineering(df)
        df = pd.merge(self.__ids, df, on=self._id_var, how='left') # add TARGET to the data (to be able to ID train/test)
        df = self._encode_data(df) # encode categorical columns for easier storing and modelling
        df = self.dataframe_objects_to_categories(df)
        X_train = df[df['TARGET'].notnull()].drop(['TARGET'], axis=1).to_records(index=False) # train contains TARGET vals
        X_test = df[df['TARGET'].isnull()].drop(['TARGET'], axis=1).to_records(index=False) # test does not
        del df # we no longer need df, let's remove it from memory
        # validate that no accidental duplicates were created by seeing that the original and current sizes are the same
        assert X_train.shape[0]==self.__ids_train.shape[0], 'Size of the train set does not correspond to the original.'
        assert X_test.shape[0]==self.__ids_test.shape[0], 'Size of the test set does not correspond to the original.'
        
        self.__export_to_numpy(X_train, X_test) 
        return [X_train, X_test] 
    
    def get_y(self):
        """Processes the dataset by applying data cleaning, feature engineering and transformation into numpy record
        arrays.

        Returns:
            Two Pandas dataframes: one containing the train set IDs and TARGET values and one containing the test set IDs.
        """
        return self.__ids_train, self.__ids_test


# In[3]:


class FeaturesApplication(ProcessData):
    """Subclass of ProcessData, used to process the Application dataset.
    """
    def __init__(self, save_to_numpy=True, random_state=404):
        super().__init__(filename='application_train.csv', save_to_numpy=save_to_numpy)
        self.__random_state = random_state
        
    def _clean_data(self, df):
        """Cleans data. This methods overrides the _clean_data method present in the parent class ProcessData.
        
        Recode/replace various values.

        Args:
            df: pandas dataframe to be cleaned.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
        df['CODE_GENDER'] = df['CODE_GENDER'].map({'F': 1, 'M': 0})
        df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map({'Y': 1, 'N': 0})
        df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map({'Y': 1, 'N': 0})
        df['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
        df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
        df['YEARS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
        return df
    
    def _feature_engineering(self, df):
        """Add features to the data. This methods overrides the _feature_engineering method present in
        the parent class ProcessData.
        
        Add new manually engineered features as well as aggregates to the data.

        Args:
            df: pandas dataframe to be used for feature engineering.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['N_EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1) 
        df['N_PMT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['N_LTV_RATIO'] = (df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']).replace(0, np.nan)
        df['N_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['N_CREDIT_TERM_YEARS'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']) / 12
        df['N_YEARS_TO_PAY_ADULT_LIFE_RATIO'] = df['N_CREDIT_TERM_YEARS'] / (df['YEARS_BIRTH'] - 18)
        
        df['N_EXT_SOURCE_MEDIAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)
        df['N_EXT_SOURCE_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
        df['N_EXT_SOURCE_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
        
        df['N_BAD_COMPANY'] = df[['DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)        
        df['N_BAD_COMPANY'] = df['N_BAD_COMPANY'].fillna(0)
        
        
        doc_flags = [col for col in df.columns if col.startswith('FLAG_DOCUMENT_')]
        df['N_DOC_FLAGS_SUM'] = df[doc_flags].sum(axis=1)
        
        # ICA doc flags
        n_pc = 3
        ica = FastICA(n_components=n_pc, random_state=self.__random_state)
        ica.fit(df[doc_flags].values)
        ica_X = ica.transform(df[doc_flags].values)
        df_ica = pd.DataFrame(data=ica_X, columns=['N_DOC_FLAGS_ICA'+str(i) for i in np.arange(1,n_pc+1)])
        df = df.assign(**df_ica)
        
        # kNN residence context (residence conditions columns + region population + housing situation)
        medi_mode_avg_cols = [col for col in df.columns if col.endswith(('_MEDI', '_MODE', '_AVG'))]
        context_cols = ['NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE']
        df_residence = df.loc[:, medi_mode_avg_cols + context_cols]
        df_residence = self._encode_data(df_residence, encoding_type='label', verbose=False)
        df_residence = df_residence.fillna(-1)
        km = KMeans(n_clusters=7, random_state=self.__random_state, n_jobs=-2)
        km.fit(df_residence.values)
        km.predict(df_residence.values)
        df['N_RESIDENCE_CLUSTERS'] = km.labels_
        
        amt_req_credit_cols = [col for col in df.columns if col.startswith('AMT_REQ_CREDIT_BUREAU_')]
        amt_req_credit_cols_month = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                                     'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON']
        df['N_AMT_REQ_CREDIT_BUREAU_SUM'] = df[amt_req_credit_cols].sum(axis=1)
        
        df['N_CHILDREN_TO_ADULT_AGE_RATIO'] = df['CNT_CHILDREN'] / (df['YEARS_BIRTH'] - 18)
        df['N_CAR_TO_ADULT_AGE_RATIO'] = df['OWN_CAR_AGE'] / (df['YEARS_BIRTH'] - 18)
        df['N_CAR_TO_TENURE_RATIO'] = df['OWN_CAR_AGE'] / df['YEARS_EMPLOYED']
        df['N_PHONE_ADULT_LIFE_RATIO'] = df['YEARS_LAST_PHONE_CHANGE'] / (df['YEARS_BIRTH'] - 18)
        df['N_CREDIT_PER_CAPITA'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
        df['N_CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / df['CNT_CHILDREN']
        df['N_CNT_ADULTS'] = df['CNT_FAM_MEMBERS'] - df['CNT_CHILDREN']
        df['N_CREDIT_PER_CAPITA'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
        df['N_CREDIT_PER_ADULT'] = df['AMT_CREDIT'] / df['N_CNT_ADULTS']
        df['N_INCOME_PER_CHILD'] =  df['AMT_INCOME_TOTAL'] / df['CNT_CHILDREN']
        df['N_INCOME_PER_ADULT'] =  df['N_CNT_ADULTS'] / df['CNT_CHILDREN']
        
        df['N_IS_SINGLE'] = ( (df['NAME_FAMILY_STATUS'] == 'Separated') | (df['NAME_FAMILY_STATUS'] == 'Widow') |
                              (df['NAME_FAMILY_STATUS'] == 'Single / not married') ).astype('int32')
        
        for i in range(1,4):
            df['N_EXT_SOURCE'+str(i)+'_DIFF'] = df['EXT_SOURCE_'+str(i)] - df['EXT_SOURCE_'+str(i)].mean()
            
        df['N_EXT_SOURCE_2_3_DIFF'] = df['EXT_SOURCE_2'] - df['EXT_SOURCE_3']
        df['N_EXT_SOURCE_3_2_DIFF'] = df['EXT_SOURCE_3'] - df['EXT_SOURCE_2']
        df['N_EXT_SOURCE_1_2_DIFF'] = df['EXT_SOURCE_1'] - df['EXT_SOURCE_2']
        
        cols_to_drop = doc_flags + medi_mode_avg_cols
        df.drop(cols_to_drop, axis=1, inplace=True)
        return df
    
class FeaturesPrevious(ProcessData):
    """Subclass of ProcessData, used to process the Previous Application dataset.
    """
    def __init__(self, save_to_numpy=True):
        super().__init__(filename='previous_application.csv', save_to_numpy=save_to_numpy)
        
    def _clean_data(self, df):
        """Cleans data. This methods overrides the _clean_data method present in the parent class ProcessData.
        
        Recode/replace various values.

        Args:
            df: pandas dataframe to be cleaned.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['CODE_REJECT_REASON'] = df['CODE_REJECT_REASON'].replace('XAP', np.nan)
        df['NAME_YIELD_GROUP'] = df['NAME_YIELD_GROUP'].map({'XNA': np.nan, 'low_action': 0, 'low_normal': 1,
                                                             'middle': 2, 'high': 3})
        return df
    
    def __get_aggregation(self, df, for_last_n_years=60):
        """Aggregates the dataset features for the specified recency.
        
        Args:
            df: pandas dataframe to be used for feature engineering.
            for_last_n_years: number of years to to be considered in the aggregation process.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df2 = df.loc[df['YEARS_DECISION']<=for_last_n_years, :] # only keep the desired timeframe
        df2 = df2.groupby(self._id_var).agg({'N_LTV_RATIO': ['mean', 'min', 'max', 'std', 'sum'],
                                             'N_PMT_RATE': ['mean', 'min', 'max', 'std', 'sum'],
                                             'N_ASK_VS_REAL': ['mean', 'min', 'max', 'std', 'sum'],
                                             'NAME_YIELD_GROUP': ['mean', 'min', 'max', 'std', 'sum'],
                                             'N_CREDIT_TERM_YEARS': ['mean', 'min', 'max', 'std', 'sum'],
                                             'N_CONTRACT_REFUSED': ['mean', 'max', 'sum'],
                                             'N_PMT_TYPE_UNKNOWN': ['mean', 'min', 'max', 'sum'],
                                             'N_YEARS_DRAWING_DECISION_DIFF': ['mean', 'min', 'max', 'std', 'sum'],
                                             'AMT_ANNUITY': ['mean', 'min', 'max', 'std', 'sum'],
                                             'AMT_APPLICATION':['mean', 'min', 'max', 'std', 'sum'],
                                             'AMT_CREDIT': ['mean', 'min', 'max', 'std', 'sum'],
                                             'AMT_DOWN_PAYMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                             'AMT_GOODS_PRICE': ['mean', 'min', 'max', 'std', 'sum'],
                                             'HOUR_APPR_PROCESS_START': ['mean', 'min', 'max', 'std', 'sum'],
                                             'RATE_DOWN_PAYMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                             'RATE_INTEREST_PRIMARY': ['mean', 'min', 'max'],
                                             'RATE_INTEREST_PRIVILEGED': ['mean', 'min', 'max'],
                                             'YEARS_DECISION': ['mean', 'min', 'max', 'std', 'sum'],
                                             'SELLERPLACE_AREA': ['mean', 'min', 'max', 'std', 'sum'],
                                             'CNT_PAYMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                             'YEARS_FIRST_DRAWING': ['mean', 'min', 'max', 'std', 'sum'],
                                             'YEARS_FIRST_DUE': ['mean', 'min', 'max', 'std', 'sum'],
                                             'YEARS_LAST_DUE_1ST_VERSION': ['mean', 'min', 'max', 'std', 'sum'],
                                             'YEARS_LAST_DUE': ['mean', 'min', 'max', 'std', 'sum'],
                                             'YEARS_TERMINATION': ['mean', 'min', 'max', 'std', 'sum']})
        df2.columns = df2.columns.map('_'.join).map(str.upper)
        df2 = df2.add_prefix('AG_PA_last'+str(for_last_n_years)+'_')
        return df2
    
    def _feature_engineering(self, df):
        """Add features to the data. This methods overrides the _feature_engineering method present in
        the parent class ProcessData.
        
        Add new manually engineered features as well as aggregates to the data.

        Args:
            df: pandas dataframe to be used for feature engineering.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['N_LTV_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
        df['N_PMT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        df['N_ASK_VS_REAL'] = df['AMT_APPLICATION'] / df['AMT_CREDIT']
        df['N_CREDIT_TERM_YEARS'] = df['CNT_PAYMENT'] / 12
        df['N_CONTRACT_REFUSED'] = (df['NAME_CONTRACT_STATUS']=='Refused').astype('int32')
        df['N_PMT_TYPE_UNKNOWN'] = (df['NAME_PAYMENT_TYPE']=='XNA').astype('int32')
        df['N_YEARS_DRAWING_DECISION_DIFF'] = df['YEARS_FIRST_DRAWING'] - df['YEARS_DECISION']
        
        df2 = self.__get_aggregation(df, for_last_n_years=3)
        df2 = pd.merge(df2, self.__get_aggregation(df, for_last_n_years=60), on='SK_ID_CURR', how='left')
        return df2
    
    
class FeaturesInstallments(ProcessData):
    """Subclass of ProcessData, used to process the Installments dataset.
    """
    def __init__(self, save_to_numpy=True):
        super().__init__(filename='installments_payments.csv', save_to_numpy=save_to_numpy)
        
    def _clean_data(self, df):
        return df
    
    def _feature_engineering(self, df):    
        """Add features to the data. This methods overrides the _feature_engineering method present in
        the parent class ProcessData.
        
        Add new manually engineered features as well as aggregates to the data.

        Args:
            df: pandas dataframe to be used for feature engineering.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['N_PMT_INSTL_RATIO'] = (df['AMT_PAYMENT'] / df['AMT_INSTALMENT']).replace([np.inf, -np.inf], np.nan)
        
        df['N_DIFF_PMT_DATE_AND_ACTUAL_PMT'] = df['YEARS_ENTRY_PAYMENT'] - df['YEARS_INSTALMENT']
        df['N_PAID_ON_TIME'] = (df['YEARS_ENTRY_PAYMENT'] == df['YEARS_INSTALMENT']).astype('int32')
        df['N_PAID_LATE'] = (df['YEARS_ENTRY_PAYMENT'] < df['YEARS_INSTALMENT']).astype('int32') 
        df['N_PAID_EARLY'] = (df['YEARS_ENTRY_PAYMENT'] > df['YEARS_INSTALMENT']).astype('int32')
        
        df2 = df.groupby('SK_ID_CURR').agg({'NUM_INSTALMENT_VERSION': ['nunique'],
                                            'N_PAID_ON_TIME': ['sum'],
                                            'N_PAID_EARLY': ['sum'],
                                            'N_PAID_LATE': ['sum'],
                                            'YEARS_INSTALMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'YEARS_ENTRY_PAYMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_INSTALMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'NUM_INSTALMENT_NUMBER': ['mean', 'min', 'max', 'std', 'sum'],
                                            'N_PMT_INSTL_RATIO': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_PAYMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'N_DIFF_PMT_DATE_AND_ACTUAL_PMT': ['mean', 'min', 'max', 'std', 'sum']})
        # the resulting data frame has column names AND labels (MultiIndex Pandas Series), we must rename accordingly
        df2.columns = df2.columns.map('_'.join).map(str.upper) # join levels with _ separator and transform to upper case
        df2 = df2.add_prefix('AG_IP_') # add prefix to all column names
        
        df2['IP_PM_COUNT'] = df2[['AG_IP_N_PAID_LATE_SUM', 'AG_IP_N_PAID_EARLY_SUM',
                                  'AG_IP_N_PAID_ON_TIME_SUM']].sum(axis=1)
        df2['IP_PAID_LATE_SHARE'] = df2['AG_IP_N_PAID_LATE_SUM'] / df2['IP_PM_COUNT']
        
        df2.drop(['IP_PM_COUNT'], axis=1, inplace=True)
        return df2
    
class FeaturesCreditCard(ProcessData):
    """Subclass of ProcessData, used to process the Credit Card dataset.
    """
    def __init__(self, save_to_numpy=True):
        super().__init__(filename='credit_card_balance.csv', save_to_numpy=save_to_numpy)
        
    def _clean_data(self, df):
        """Cleans data. This methods overrides the _clean_data method present in the parent class ProcessData.
        
        Recode/replace various values.

        Args:
            df: pandas dataframe to be cleaned.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df.loc[df['AMT_DRAWINGS_ATM_CURRENT'] < 0, 'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
        df.loc[df['AMT_DRAWINGS_CURRENT'] < 0, 'AMT_DRAWINGS_CURRENT'] = np.nan
        return df
    
    def __get_max_load(self, df):
        """Get the maximum load, defined as the maximum credit utilization, for each credit card.

        Args:
            df: pandas dataframe to be used for feature engineering.

        Returns:
            Pandas dataframe containing the results.
        """
        df2 = df.groupby(['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
            lambda g: g['AMT_BALANCE'].max() / g['AMT_CREDIT_LIMIT_ACTUAL'].max() ).reset_index()
        df2.rename(columns={0: 'MAX_LOAD'}, inplace=True)
        return df2
    
    def __get_util_for_month(self, df, for_month):
        """Get load for the specified month.

        Args:
            df: pandas dataframe to be used for feature engineering.
            for_month: month to get the credit utilization for.

        Returns:
            Pandas dataframe containing the results.
        """
        df_active_month = df.loc[(df['MONTHS_BALANCE']==for_month) & (df['NAME_CONTRACT_STATUS']=='Active'), :]
        df_agg = df_active_month.groupby('SK_ID_CURR').agg({'AMT_BALANCE': 'sum', 'AMT_CREDIT_LIMIT_ACTUAL': 'sum'})
        df_agg['N_UTIL'] = np.clip(df_agg['AMT_BALANCE'] / df_agg['AMT_CREDIT_LIMIT_ACTUAL'], 0, 1)
        df_agg.drop(['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL'], axis=1, inplace=True)
        df_agg = df_agg.add_prefix('N_CC_MONTH'+str(-for_month)+'_')
        return df_agg
    
    def __get_util_trend_for_last_n_months(self, df, n_months):
        """Get the trendline slope for the credit utilization of the last n specified months.

        Args:
            df: pandas dataframe to be used for feature engineering.
            n_months: number of months to get the credit utilization trendline for.

        Returns:
            Pandas dataframe containing the results.
        """
        df_months = df.loc[df['MONTHS_BALANCE']>=-n_months, :]
        df_agg = df_months.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg({
                                                'AMT_BALANCE': 'sum', 'AMT_CREDIT_LIMIT_ACTUAL': 'sum'})
        df_agg['N_UTIL'] = np.clip(df_agg['AMT_BALANCE'] / df_agg['AMT_CREDIT_LIMIT_ACTUAL'], 0, 1)
        df_agg.drop(['AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL'], axis=1, inplace=True)
        df_agg = df_agg.dropna().reset_index()
        
        df_agg2 = df_agg.groupby('SK_ID_CURR')            .apply(lambda g: self._trendline(data=g['N_UTIL'], y=g['MONTHS_BALANCE']))            .reset_index(name='N_CC_UTIL_TREND'+str(n_months))
        return df_agg2
    
    def __get_trend_for_last_n_months(self, df, column, n_months):
        """Get the trendline slope of the last n specified months for the desired variable.

        Args:
            df: pandas dataframe to be used for feature engineering.
            column: variable to use when computing the trendline.
            n_months: number of months to get the trendline for.

        Returns:
            Pandas dataframe containing the results.
        """
        df_months = df.loc[df['MONTHS_BALANCE']>=-n_months, :]
        df_agg = df_months.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg({column: 'sum'})
        df_agg = df_agg.dropna().reset_index()
        
        df_agg2 = df_agg.groupby('SK_ID_CURR')            .apply(lambda g: self._trendline(data=g[column], y=g['MONTHS_BALANCE']))            .reset_index(name='N_CC_'+column+'_TREND'+str(n_months))
        return df_agg2
    
    def _feature_engineering(self, df):        
        """Add features to the data. This methods overrides the _feature_engineering method present in
        the parent class ProcessData.
        
        Add new manually engineered features as well as aggregates to the data.

        Args:
            df: pandas dataframe to be used for feature engineering.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['DPD_GAP'] = df['SK_DPD'] - df['SK_DPD_DEF']
        
        df2 = self.__get_util_for_month(df, -1)
        df2 = pd.merge(df2, self.__get_util_trend_for_last_n_months(df, n_months=7), on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_util_trend_for_last_n_months(df, n_months=60), on='SK_ID_CURR', how='left')
        
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'AMT_RECEIVABLE_PRINCIPAL', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'AMT_BALANCE', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'AMT_CREDIT_LIMIT_ACTUAL', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'AMT_DRAWINGS_ATM_CURRENT', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'CNT_DRAWINGS_CURRENT', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'CNT_DRAWINGS_ATM_CURRENT', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'AMT_PAYMENT_TOTAL_CURRENT', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'SK_DPD', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'SK_DPD_DEF', n_months=60),
                       on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'DPD_GAP', n_months=60),
                       on='SK_ID_CURR', how='left')
        
        df = pd.merge(df, self.__get_max_load(df), on=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL'])
        df['SHARE_PAID'] = df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_TOTAL_RECEIVABLE']
        df['CASH_SHARE'] = df['AMT_DRAWINGS_ATM_CURRENT'] / df['AMT_BALANCE']
        df['CONTRACT_COMPLETED'] = (df['NAME_CONTRACT_STATUS'] == 'Completed').astype(int)
        df['DPD_ON_FILE'] = (df['SK_DPD'] > 0).astype(int)
        df['DPD_DEF_ON_FILE'] = (df['SK_DPD_DEF'] > 0).astype(int)
        
        
        df3 = df.groupby('SK_ID_CURR').agg({'MAX_LOAD': ['mean', 'min', 'max', 'std', 'sum'],
                                            'SK_DPD_DEF': ['max'],
                                            'SK_DPD': ['max'],
                                            'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'SHARE_PAID': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CASH_SHARE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_BALANCE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_DRAWINGS_CURRENT': ['mean', 'min', 'max', 'std', 'sum', 'skew'],
                                            'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'max', 'std', 'sum', 'count'],
                                            'AMT_DRAWINGS_OTHER_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_DRAWINGS_POS_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_INST_MIN_REGULARITY': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_PAYMENT_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_PAYMENT_TOTAL_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_RECIVABLE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_TOTAL_RECEIVABLE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CNT_DRAWINGS_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CNT_DRAWINGS_OTHER_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CNT_DRAWINGS_POS_CURRENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CNT_INSTALMENT_MATURE_CUM': ['mean', 'min', 'max', 'std', 'sum'],
                                            'SK_DPD': ['mean', 'max', 'std', 'sum', 'count'],
                                            'SK_DPD_DEF': ['mean', 'max', 'std', 'sum', 'count'],
                                            'MONTHS_BALANCE': ['count'],
                                            'DPD_ON_FILE': ['mean', 'sum'],
                                            'DPD_DEF_ON_FILE': ['mean', 'sum'],
                                            'DPD_GAP': ['mean'],
                                            'CONTRACT_COMPLETED': ['mean']})
        
        df3.columns = df3.columns.map('_'.join).map(str.upper) # join levels with _ separator and transform to upper case
        df3 = df3.add_prefix('AG_CC_') # add prefix to all column names
        
        df2 = pd.merge(df2, df3, on='SK_ID_CURR', how='left')
        
        df2['N_CC_MONTHS_BEFORE_MAXING_OUT7'] = (1 - df2['N_CC_MONTH1_N_UTIL']) / df2['N_CC_UTIL_TREND7']
        df2['N_CC_MONTHS_BEFORE_MAXING_OUT7'] = df2['N_CC_MONTHS_BEFORE_MAXING_OUT7'].replace(np.inf, 1000)
        
        df2['N_CC_TREND7_60_RATIO'] = df2['N_CC_UTIL_TREND7'] / df2['N_CC_UTIL_TREND60']
        df2['N_CC_TREND7_60_DIFF'] = df2['N_CC_UTIL_TREND7'] - df2['N_CC_UTIL_TREND60']
        
        df2['N_CC_AMT_RECEIVABLE_PRINCIPAL_TREND60_DIFF'] = df2['N_CC_AMT_RECEIVABLE_PRINCIPAL_TREND60'] -                                                        df2['N_CC_AMT_RECEIVABLE_PRINCIPAL_TREND60'].median()
        
        df2['N_CC_LIMIT_GROWING_SLOWER_THAN_BALANCE'] =                                 (df2['N_CC_AMT_CREDIT_LIMIT_ACTUAL_TREND60'] < df2['N_CC_AMT_BALANCE_TREND60']).astype(int)
        return df2    
    
class FeaturesPOS(ProcessData):
    """Subclass of ProcessData, used to process the Point of Sales dataset.
    """
    def __init__(self, save_to_numpy=True):
        super().__init__(filename='POS_CASH_balance.csv', save_to_numpy=save_to_numpy)
        
    def _clean_data(self, df):
        return df
    
    # TODO: refactoring, integrate into parent class
    def __get_trend_for_last_n_months(self, df, column, n_months):
        """Get the trendline slope of the last n specified months for the desired variable.

        Args:
            df: pandas dataframe to be used for feature engineering.
            column: variable to use when computing the trendline.
            n_months: number of months to get the trendline for.

        Returns:
            Pandas dataframe containing the results.
        """
        df_months = df.loc[df['MONTHS_BALANCE']>=-n_months, :]
        df_agg = df_months.groupby(['SK_ID_CURR', 'MONTHS_BALANCE']).agg({column: 'sum'})
        df_agg = df_agg.dropna().reset_index()
        
        df_agg2 = df_agg.groupby('SK_ID_CURR')            .apply(lambda g: self._trendline(data=g[column], y=g['MONTHS_BALANCE']))            .reset_index(name='N_CC_'+column+'_TREND'+str(n_months))
        return df_agg2
    
    def _feature_engineering(self, df):        
        """Add features to the data. This methods overrides the _feature_engineering method present in
        the parent class ProcessData.
        
        Add new manually engineered features as well as aggregates to the data.

        Args:
            df: pandas dataframe to be used for feature engineering.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['CREDIT_TERM_YEARS'] = df['CNT_INSTALMENT'] / 12
        df['ELAPSED_TERM'] = df['CNT_INSTALMENT_FUTURE'] / df['CNT_INSTALMENT']
        df['CONTRACT_COMPLETED'] = (df['NAME_CONTRACT_STATUS']=='Completed').astype('int32')
        df['DPD_ON_FILE'] = (df['SK_DPD'] > 0).astype('int32')
        df['DPD_DEF_ON_FILE'] = (df['SK_DPD_DEF'] > 0).astype('int32')
        df['DPD_GAP'] = df['SK_DPD'] - df['SK_DPD_DEF']

        df2 = df.groupby('SK_ID_CURR').agg({'MONTHS_BALANCE': ['mean', 'min', 'max', 'std', 'sum', 'count'],
                                            'CNT_INSTALMENT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CNT_INSTALMENT_FUTURE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'SK_DPD_DEF': ['mean', 'min', 'max', 'std', 'sum'],
                                            'DPD_GAP': ['mean', 'min', 'max', 'std', 'sum'],
                                            'MONTHS_BALANCE': ['count', 'min'],
                                            'DPD_ON_FILE': ['mean', 'sum'],
                                            'DPD_DEF_ON_FILE': ['mean', 'sum'],
                                            'DPD_GAP': ['mean', 'max', 'std', 'sum'],
                                            'CONTRACT_COMPLETED': ['mean']})
        # the resulting data frame has column names AND labels (MultiIndex Pandas Series), we must rename accordingly
        df2.columns = df2.columns.map('_'.join).map(str.upper) # join levels with _ separator and transform to upper case
        df2 = df2.add_prefix('AG_POS_') # add prefix to all column names
        
        df_last_month = df.loc[df['MONTHS_BALANCE']==-1, :]
        df3 = df_last_month.groupby('SK_ID_CURR').agg({'CREDIT_TERM_YEARS': ['mean', 'min', 'max', 'std', 'sum'],
                                              'ELAPSED_TERM': ['mean', 'min', 'max', 'std', 'sum'],
                                              'SK_DPD': ['max'],
                                              'CNT_INSTALMENT_FUTURE': ['max']})
        df3.columns = df3.columns.map('_'.join).map(str.upper) # join levels with _ separator and transform to upper case
        df3 = df3.add_prefix('AG_POSm1_') # add prefix to all column names
        df2 = pd.merge(df3, df2, on='SK_ID_CURR', how='left')
        
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'SK_DPD', n_months=60), on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'SK_DPD', n_months=10), on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'SK_DPD_DEF', n_months=60), on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'SK_DPD_DEF', n_months=10), on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'DPD_GAP', n_months=60), on='SK_ID_CURR', how='left')
        df2 = pd.merge(df2, self.__get_trend_for_last_n_months(df, 'DPD_GAP', n_months=10), on='SK_ID_CURR', how='left')
        return df2
    
class FeaturesBureau(ProcessData):
    """Subclass of ProcessData, used to process the Credit Bureau dataset.
    """
    def __init__(self, save_to_numpy=True):
        super().__init__(filename='bureau.csv', save_to_numpy=save_to_numpy)
        self.df_bal = self.get_csv_or_hd5(os.path.join(self._input_path, 'bureau_balance.csv'))
        
    def _clean_data(self, df):
        """Cleans data. This methods overrides the _clean_data method present in the parent class ProcessData.
        
        Recode/replace various values.

        Args:
            df: pandas dataframe to be cleaned.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df.loc[df['YEARS_ENDDATE_FACT'] < -40000, 'YEARS_ENDDATE_FACT'] = np.nan
        df.loc[df['YEARS_CREDIT_UPDATE'] < -40000, 'YEARS_CREDIT_UPDATE'] = np.nan
        df.loc[df['YEARS_CREDIT_ENDDATE'] < -40000, 'YEARS_CREDIT_ENDDATE'] = np.nan
        
        df[['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE', 'CNT_CREDIT_PROLONG']].fillna(0,
                                                                                                             inplace=True)
        df.drop(['CREDIT_CURRENCY'], axis=1, inplace=True) # almost entirely made up of CURRENCY_1... let's drop it
        return df
    
    def __get_util(self, df, credit_type='All', credit_status='Active'):
        """Get credit utilization.

        Args:
            df: pandas dataframe to be used for feature engineering.
            credit_type: credit type to get the credit utilization for.
            credit_status: credit status to get the credit utilization for.

        Returns:
            Pandas dataframe containing the results.
        """
        limit_column = 'AMT_CREDIT_SUM'
        if credit_type != 'All':
            df = df.loc[df['CREDIT_ACTIVE']==credit_status, :]
        if credit_type != 'All':
            df = df.loc[df['CREDIT_TYPE']==credit_type, :]
        if credit_type == 'Credit card':
            limit_column = 'AMT_CREDIT_SUM_LIMIT'
        
        df_agg = df.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM_DEBT': 'sum', limit_column: 'sum'})
        df_agg['N_UTIL'] = np.clip(df_agg['AMT_CREDIT_SUM_DEBT'] / df_agg[limit_column], 0, 1)
        df_agg.drop(['AMT_CREDIT_SUM_DEBT', limit_column], axis=1, inplace=True)
        df_agg = df_agg.add_prefix('N_BU_UTIL_'+str(credit_status.upper())+'_'+str(credit_type.upper())+'_')
        return df_agg
    
    def __get_count(self, df, credit_type='All', credit_status='Active'):
        """Get counts for credit types and status.

        Args:
            df: pandas dataframe to be used for feature engineering.
            credit_type: credit type to get the count for.
            credit_status: credit status to get the count for.

        Returns:
            Pandas dataframe containing the results.
        """
        if credit_type != 'All':
            df = df.loc[df['CREDIT_ACTIVE']==credit_status, :]
        if credit_type != 'All':
            df = df.loc[df['CREDIT_TYPE']==credit_type, :]
        
        df_agg = df.groupby('SK_ID_CURR').size().reset_index(
            name='N_BU_COUNT_'+str(credit_status.upper())+'_'+str(credit_type.upper()))
        df_agg.columns = df_agg.columns.str.replace(' ', '_') # some credit types have whitespaces, let's replace them
        return df_agg
    
    def __get_stats_dpd(self, df, n_months, stats, status_cutoff=None):
        """Get aggregates for Days Past Due (DPD), also known as credit delinquencies.

        Args:
            df: pandas dataframe to be used for feature engineering.
            n_months: number of months to take into consideration for computing the DPD statistics.
            stats: statistics to use at the time of aggregation (e.g. mean, std)
            status_cutoff: threshold for the type of credit delinquency. It can be used to exclude
            non-delinquencies, for example.

        Returns:
            Pandas dataframe containing the results.
        """
        df_hist = df.loc[(df['STATUS']!='X') & (df['STATUS']!='C') & (df['MONTHS_BALANCE']>=-n_months),
                         ['SK_ID_CURR', 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']]
        df_hist['STATUS'] = df_hist['STATUS'].astype('int32') 
        
        if status_cutoff != None:
            df_hist = df_hist.loc[df_hist['STATUS']>=status_cutoff,
                                  ['SK_ID_CURR', 'SK_ID_BUREAU', 'MONTHS_BALANCE', 'STATUS']]
        
        df2 = df_hist.groupby('SK_ID_CURR').agg({'STATUS': stats})
        df2.columns = df2.columns.map('_'.join).map(str.upper)
        df2 = df2.add_prefix('AG_BU_DPD_LAST'+str(n_months)+'_')
        return df2
    
    def __get_history_status_counts(self, df, for_last_n_months=None):
        """Get counts for types of credit delinquencies. See parent class method _count_categories_pivot
        for more information on the how the counting process works.

        Args:
            df: pandas dataframe to be used for feature engineering.
            for_last_n_months: number of months to take into consideration for computing the counts.

        Returns:
            Pandas dataframe containing the results.
        """
        if for_last_n_months == None:
            return self._count_categories_pivot(df, 'STATUS', name='BU_HISTORY', by_id='SK_ID_BUREAU')
        else:
            df = df.loc[df['MONTHS_BALANCE'] >= -for_last_n_months, :]
            return self._count_categories_pivot(df, 'STATUS', name='BU_HISTORY_LAST'+str(for_last_n_months),
                                                      by_id='SK_ID_BUREAU')
        
    def __get_trend_for_last_n_months(self, df, n_months, replace_x=None):
        """Get the trendline slope of the last n specified months for delinquency types.

        Args:
            df: pandas dataframe to be used for feature engineering.
            n_months: number of months to get the trendline for.
            replace_x: what value to replace X (i.e. unknown delinquency status) by before computing
            the trend.

        Returns:
            Pandas dataframe containing the results.
        """
        if replace_x == None:
            df = df.loc[(df['STATUS']!='X') & (df['STATUS']!='C'), :]
        else:
            df['STATUS'].replace('X', replace_x, inplace=True)
            df = df.loc[df['STATUS']!='C', :]
            
        df = df.loc[df['MONTHS_BALANCE']>=-n_months, :]
        df['STATUS'] = df['STATUS'].astype('int32') 
        
        df_agg = df.groupby('SK_ID_BUREAU')            .apply(lambda g: self._trendline(data=g['STATUS'], y=g['MONTHS_BALANCE']))            .reset_index(name='N_BU_STATUS_TREND'+str(n_months))
        return df_agg
    
    def _feature_engineering(self, df):
        """Add features to the data. This methods overrides the _feature_engineering method present in
        the parent class ProcessData.
        
        Add new manually engineered features as well as aggregates to the data.

        Args:
            df: pandas dataframe to be used for feature engineering.

        Returns:
            Pandas dataframe containing the modified data.
        """
        df['NEW_BU_ACTIVE_CREDIT'] = (df['CREDIT_ACTIVE'] != 'Closed').astype('int32')
        df['NEW_BU_POSITIVE_ENDDATE'] = (df['YEARS_CREDIT_ENDDATE'] > 0).astype('int32')
        
        df2 = df.groupby('SK_ID_CURR').agg({'YEARS_CREDIT': ['count', 'mean', 'min', 'max', 'std'],
                                            'CREDIT_DAY_OVERDUE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'YEARS_ENDDATE_FACT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CNT_CREDIT_PROLONG': ['mean', 'min', 'max', 'std', 'sum'],
                                            'YEARS_CREDIT_ENDDATE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_CREDIT_SUM_DEBT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'AMT_ANNUITY': ['mean', 'min', 'max', 'std', 'sum'],
                                            'NEW_BU_ACTIVE_CREDIT': ['mean', 'min', 'max', 'std', 'sum'],
                                            'NEW_BU_POSITIVE_ENDDATE': ['mean', 'min', 'max', 'std', 'sum'],
                                            'CREDIT_TYPE': ['nunique']})
        df2.columns = df2.columns.map('_'.join).map(str.upper)
        df2 = df2.add_prefix('AG_BUm1_')
        
        df2['N_BU_SHARE_OVERDUE'] = df2['AG_BUm1_AMT_CREDIT_SUM_OVERDUE_SUM'] / df2['AG_BUm1_AMT_CREDIT_SUM_DEBT_SUM']
        df2['AG_BUm1_PAST_LOANS_PER_TYPE'] = df2['AG_BUm1_YEARS_CREDIT_COUNT'] / df2['AG_BUm1_CREDIT_TYPE_NUNIQUE']
        
        for t in ['All', 'Credit card']:
            df2 = pd.merge(df2, self.__get_util(df, t, 'Active'), on='SK_ID_CURR', how='left')
                                   
        for t in ['Credit card', 'Car loan', 'Mortgage', 'Consumer credit']:
            df2 = pd.merge(df2, self.__get_count(df, t, 'Active'), on='SK_ID_CURR', how='left')
        
        df2['N_BU_PMT_RATE'] = df2['AG_BUm1_AMT_ANNUITY_SUM'] / df2['AG_BUm1_AMT_CREDIT_SUM_DEBT_SUM']
        
        ### now historical data
        for t in [9, 24, 96]:
            df = pd.merge(df, self.__get_history_status_counts(self.df_bal, for_last_n_months=t),
                          on='SK_ID_BUREAU', how='left')
            df3 = df.groupby('SK_ID_CURR').agg(
                {'N_BU_HISTORY_LAST'+str(t)+'_STATUS_2_COUNT': ['count', 'mean', 'max', 'std'],
                 'N_BU_HISTORY_LAST'+str(t)+'_STATUS_3_COUNT': ['count', 'mean', 'max', 'std'],
                 'N_BU_HISTORY_LAST'+str(t)+'_STATUS_4_COUNT': ['count', 'mean', 'max', 'std'],
                 'N_BU_HISTORY_LAST'+str(t)+'_STATUS_5_COUNT': ['count', 'mean', 'max', 'std']})
            df3.columns = df3.columns.map('_'.join).map(str.upper)
            df2 = pd.merge(df3, df2, on='SK_ID_CURR', how='left')
            
        for t in [6, 18]:
            df = pd.merge(df, self.__get_history_status_counts(self.df_bal, for_last_n_months=t),
                          on='SK_ID_BUREAU', how='left')
            df3 = df.groupby('SK_ID_CURR').agg({'N_BU_HISTORY_LAST'+str(t)+'_STATUS_X_COUNT': ['count', 'mean', 'max', 'std']})
            df3.columns = df3.columns.map('_'.join).map(str.upper)
            df2 = pd.merge(df3, df2, on='SK_ID_CURR', how='left')
        
        for t in [6, 12, 24, 60]:
            df = pd.merge(df, self.__get_trend_for_last_n_months(self.df_bal, t), on='SK_ID_BUREAU', how='left')
            df3 = df.groupby('SK_ID_CURR').agg({'N_BU_STATUS_TREND'+str(t): ['mean', 'max', 'std']})
            df3.columns = df3.columns.map('_'.join).map(str.upper)
            df2 = pd.merge(df3, df2, on='SK_ID_CURR', how='left')
            
        df2['N_BU_STATUS_TREND_6_12_RATIO'] = df2['N_BU_STATUS_TREND6_MEAN'] / df2['N_BU_STATUS_TREND12_MEAN']
        df2['N_BU_STATUS_TREND_12_24_RATIO'] = df2['N_BU_STATUS_TREND12_MEAN'] / df2['N_BU_STATUS_TREND24_MEAN']
        df2['N_BU_STATUS_TREND_12_60_RATIO'] = df2['N_BU_STATUS_TREND12_MEAN'] / df2['N_BU_STATUS_TREND60_MEAN']
        df2['N_BU_STATUS_TREND_6_12_DIFF'] = df2['N_BU_STATUS_TREND6_MEAN'] - df2['N_BU_STATUS_TREND12_MEAN']
        df2['N_BU_STATUS_TREND_12_24_DIFF'] = df2['N_BU_STATUS_TREND12_MEAN'] - df2['N_BU_STATUS_TREND24_MEAN']
        return df2


# In[4]:


class HomeCredit:
    """Class for handling the HomeCredit model training/validation as well as data loading/processing.
    """
    def __init__(self, num_folds=5, random_state=404, stratified_kfold=True):
        self.output_path = '../output/'
        self.random_state = random_state
        self.num_folds = num_folds
        self.stratified_kfold = stratified_kfold
    
    def load_all(self, dict_data_and_numpy):
        """Loads all the HomeCredit data.

        Args:
            dict_data_and_numpy: dictionary containing dataset names as keys and a boolean value to indicate
            whether to load the correspondent dataset from the Numpy array. If it is False, the dataset is loaded
            and processed from scratch. Usually, False would come after having edited the data processing (feature
            engineering) methods for the dataset.
        """
        loaded_data = []
        app = FeaturesApplication(random_state=self.random_state)
        for dataset, load_from_numpy in dict_data_and_numpy.items():
            if dataset=='application':
                loaded_data.append(app.load_data(load_from_numpy))
            elif dataset=='previous':
                loaded_data.append(FeaturesPrevious().load_data(load_from_numpy))
            elif dataset=='installments':
                loaded_data.append(FeaturesInstallments().load_data(load_from_numpy))
            elif dataset=='card':
                loaded_data.append(FeaturesCreditCard().load_data(load_from_numpy))
            elif dataset=='pos':
                loaded_data.append(FeaturesPOS().load_data(load_from_numpy))
            elif dataset=='bureau':
                loaded_data.append(FeaturesBureau().load_data(load_from_numpy))
                
        # merge all the train and set datasets using reduce if more than one dataset, otherwise just convert to df
        if len(loaded_data) > 1:
            df_train =reduce(lambda left,right: pd.merge(pd.DataFrame.from_records(left), pd.DataFrame.from_records(right),
                                on='SK_ID_CURR', how='left'), [train[0] for train in loaded_data])
            df_test = reduce(lambda left,right: pd.merge(pd.DataFrame.from_records(left), pd.DataFrame.from_records(right),
                                on='SK_ID_CURR', how='left'), [test[1] for test in loaded_data])
        else:
            df_train = pd.DataFrame.from_records(loaded_data[0][0])
            df_test = pd.DataFrame.from_records(loaded_data[0][1])
            
        self.df_y, self.submission_df = app.get_y()
        
        assert all(self.df_y['SK_ID_CURR']==df_train['SK_ID_CURR']), 'Order of IDs in X and y do not match.'
        self.df_train, self.df_test = self.__feature_engineering(df_train, df_test) # feature engineering with all data
        assert self.df_train.shape[1]==self.df_test.shape[1], 'Train and test sets have a different number of features.'
        
        self.df_train.drop(['SK_ID_CURR'], axis=1, inplace=True) # remove SK_ID_CURR from train set
        self.df_test.drop(['SK_ID_CURR'], axis=1, inplace=True) # remove SK_ID_CURR from test set
        self.feature_names = self.df_train.columns # save column names for the feature_importance_df
        print('[!] Train: {} ; Test: {}'.format(self.df_train.shape, self.df_test.shape))
    
    def __get_folds(self):
        """Gets train and test indices for K-Fold cross-validation.
        """
        if self.stratified_kfold:
            folds = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        else:
            folds = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)
        return folds
    
    def __set_proper_model_random_state(self, model_function, model_params):
        """Sets the random state or seed for the model in the model parameters.

        Args:
            model_function: function of the model to be used (e.g. LGBMClassifier). The function must
            have either random_state or seed in the argument list.
            model_params: dictionary with the model parameters.
        
        Returns:
            model_params: dictionary of model parameters integrating the proper seed/state name.
        """
        model_arg_list = inspect.getfullargspec(model_function).args
        # see if passed model takes random_state OR seed
        if 'random_state' in model_arg_list: # sklearn models and LightGBM
            model_params['random_state'] = self.random_state
        elif 'seed' in model_arg_list: # other models such as XGBoost
            model_params['seed'] = self.random_state
        return model_params
    
    def __create_model_output(self, output_string, in_fold_preds, out_of_fold_preds, test_preds, fold_val_auc,
                              model_start_time, train_folds):
        """Creates the output string for the model for logging purposes.

        Args:
            output_string: string to add logging/output to.
            in_fold_preds: in-fold prediction array.
            out_of_fold_preds: out-of-fold prediction array.
            test_preds: array with predictions obtained using the test dataset.
            fold_val_auc: Area Under the ROC curve for the fold.
            model_start_time: time.time() object created at the start of training.
            train_folds: list containing the fold numbers being used for training.
        
        Returns:
            output_string: new output_string with all the new information appended to the input output_string.
        """
        full_val_auc = round(np.mean(fold_val_auc), 5)
        if len(train_folds)==0:
            train_folds = [[i for i in range(1,len(fold_val_auc)+1)]]
        print("-"*71)
        output_string += "[!]  Full - Train AUC: {0:.4f} ; Validation AUC: {1:.4f} ; Time: {2:.1f}s\n".format(            roc_auc_score(self.df_y['TARGET'].values, in_fold_preds), full_val_auc, (time.time() - model_start_time))
        output_string += "folds: " + str(train_folds) + "\nstratified: " + str(self.stratified_kfold) 
        output_string += "\nrandom_state: " + str(self.random_state) + "\nCV AUC validation std: "
        output_string += str(round(np.array(fold_val_auc).std(ddof=1), 5)) # compute sample standard deviation
        print(output_string)
        print(self.model[0])
            
        self.submission_df['TARGET'] = test_preds
        self.out_file_name = self.output_path + "sub_" + str(full_val_auc) + "_" + datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        self.submission_df.to_csv(self.out_file_name + ".csv", index = False)
        print(output_string, file = open(self.out_file_name + ".txt", "a"))
        print(self.model[0], file = open(self.out_file_name + ".txt", "a"))
        return output_string
    
    def __train_fold(self, model, train_index, val_index, in_fold_preds, out_of_fold_preds, test_preds, fold_val_auc,
                      output_string, fit_params, n_fold, shuffle_y=False):
        """Trains the specified fold for the model.

        Args:
            model: model function.
            train_index: train index for partitioning the data.
            val_index: validation index for partitioning the data.
            in_fold_preds: in-fold prediction array.
            out_of_fold_preds: out-of-fold prediction array.
            test_preds: array with predictions obtained using the test dataset.
            fold_val_auc: Area Under the ROC curve for the fold.
            output_string: string that serves as log.
            fit_params: parameters used for the model.fit() method.
            n_fold: number of the fold to train (e.g. 3).
            shuffle_y: boolean to shuffle the target value, combined with other methods, it can be used as a way to assess 
            feature importance.
        
        Returns:
            in_fold_preds: updated array.
            out_of_fold_preds: updated array.
            test_preds: updated array.
            fold_val_auc: updated list.
            output_string: updated string.
        """
        start_time = time.time()
        X_train, X_val = self.df_train.values[train_index], self.df_train.values[val_index]
        y_train, y_val = self.df_y['TARGET'].values[train_index], self.df_y['TARGET'].values[val_index]
        
        if shuffle_y: # for feature selection purposes
            np.random.shuffle(y_train)
            np.random.shuffle(y_val)
        
        if fit_params != None: # (e.g. LightGBM)
            model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], **fit_params)
        else:
            model.fit(X_train, y_train)
            
        # predict for train and validation sets as well as for the test set
        out_of_fold_preds[val_index] = model.predict_proba(X_val)[:,1]
        in_fold_preds[train_index] = model.predict_proba(X_train)[0:len(train_index),1]
        test_preds += model.predict_proba(self.df_test.values, num_iteration=model.best_iteration_)[:,1] / self.num_folds
        fold_val_auc.append(roc_auc_score(y_val, out_of_fold_preds[val_index]))
            
        fold_string = " Fold {0}/{1} - Train AUC: {2:.4f} ; Validation AUC: {3:.4f} ; Time: {4:.1f}s"                 .format(n_fold+1, self.num_folds, roc_auc_score(y_train[0:len(train_index)], in_fold_preds[train_index]),
                        roc_auc_score(y_val, out_of_fold_preds[val_index]), (time.time() - start_time))
        print(fold_string)
        output_string += fold_string + '\n'
        self.model.append(model)
        return in_fold_preds, out_of_fold_preds, test_preds, fold_val_auc, output_string
    
    def run_model_cv(self, model_function, model_params, fit_params=None, debug=False, train_folds=[], shuffle_y=False):
        """Trains the specified fold for the model.

        Args:
            model_function: model function to use for training.
            model_params: dictionary of parameters for the model.
            fit_params: parameters used for the model.fit() method.
            debug: boolean. There is not logging output saved if True.
            train_folds: list of folds to train (e.g. [1,2,3]). Empty is used to train for all folds.
            shuffle_y: boolean to shuffle the target value, combined with other methods, it can be used as a way to assess 
            feature importance.
        """
        model_start_time = time.time()
        out_of_fold_preds = in_fold_preds = np.zeros(self.df_train.shape[0])
        test_preds = np.zeros(self.df_test.shape[0]) # for submission
        self.feature_importance_df = pd.DataFrame()
        fold_val_auc = []
        self.model = []
        output_string = ''
        folds = self.__get_folds()
        model_params = self.__set_proper_model_random_state(model_function, model_params)
        
        model = model_function(**model_params)
        print("[*] Training model...")
        
        for n_fold, (train_index, val_index) in enumerate(folds.split(self.df_train.values, self.df_y['TARGET'].values)):
            if len(train_folds) > 0:
                if (n_fold + 1) not in train_folds:
                    print(' Skipping fold: {}...'.format((n_fold + 1)))
                    continue
                    
            in_fold_preds, out_of_fold_preds, test_preds, fold_val_auc, output_string =                     self.__train_fold(model, train_index, val_index, in_fold_preds, out_of_fold_preds, test_preds,
                                       fold_val_auc, output_string, fit_params, n_fold, shuffle_y)
            # capture feature importance by fold
            try:
                self.feature_importance_df = pd.concat([self.feature_importance_df, pd.DataFrame({'feature':
                        self.feature_names, 'importance':model.feature_importances_, 'fold':(n_fold + 1)})], axis = 0)
            except:
                pass
            
        self.val_preds = out_of_fold_preds
        
        if not debug: # not debugging, let's write output
            self.__create_model_output(output_string, in_fold_preds, out_of_fold_preds, test_preds, fold_val_auc,
                                        model_start_time, train_folds)
        
    def print_feature_importance(self):
        """Shows the feature importance mean and standard deviation.
        
        It can only be used after training the model. Results are also saved to a text file for later analysis.
        """
        df = self.feature_importance_df.groupby('feature').agg({'importance': ['mean','std'] })
        df.columns = df.columns.map('_'.join).map(str.upper)
        df = df.sort_values(by='IMPORTANCE_MEAN', ascending=False)
        try:
            print(df.to_string(), file = open(self.out_file_name + ".fi.txt", "a"))
        except:
            pass
        print(df.to_string())          
    
    @staticmethod
    def group_aggregation(df, group_vars, by_var, stats=['mean'], add_diff=False, add_z_score=False, add_abs_diff=False):
        """Add a group of statistics by group of variables and merge it to the original dataframe
        
        Compiles several aggregate statistics by group using the groupby and aggregate methods of Pandas dataframes.
        It can also add Z score and difference from the mean.
        
        Args:
            df: pandas dataframe to compute the stats with.
            group_vars: list of statistics to use for aggregation purposes.
            by_var: string name of the variable to use in the statistical aggregation.
            stats: list of statistics to use for aggregation purposes.
            add_diff: boolean on whether to add the difference from the mean or not.
            add_z_score: boolean on whether to add the z score or not.
            
        Returns:
            A dataframe with all the requested aggregations merged to the original dataframe (df) passed as
            an argument.
        """        
        if isinstance(stats, str): # in case something like 'mean' is passed instead of ['mean']
            stats = [stats] 
        if isinstance(group_vars, str): # in case something like 'OCCUPATION_TYPE' is passed instead of ['OCCUPATION_TYPE']
            group_vars = [group_vars] 
        group_agg_df = df[group_vars + [by_var]].groupby(group_vars).agg({by_var: stats})
        group_agg_df.columns = group_agg_df.columns.map('_'.join).map(str.upper)
        group_agg_df = group_agg_df.add_prefix('Gr_' + 'n'.join(group_vars) + '___')
        
        if add_diff or add_abs_diff or add_z_score:
            df2 = pd.merge(df.loc[:, group_vars + [by_var, 'SK_ID_CURR'] ], group_agg_df, how='left', on=group_vars)
            if add_diff:
                if sum([x in ['mean'] for x in stats]) == 1:
                    mean_name = [col for col in group_agg_df.columns if col.endswith('_MEAN')][0]
                    diff_name = mean_name.replace('_MEAN', '_DIFF')
                    if mean_name in df2.columns:
                        df2[diff_name] = df2[by_var] - df2[mean_name]
                else:
                    print('[!] Cannot add diff: mean is missing.')
            if add_abs_diff:
                if sum([x in ['mean'] for x in stats]) == 1:
                    mean_name = [col for col in group_agg_df.columns if col.endswith('_MEAN')][0]
                    abs_diff_name = mean_name.replace('_MEAN', '_absDIFF')
                    if mean_name in df2.columns:
                        df2[abs_diff_name] = np.abs(df2[by_var] - df2[mean_name])
                else:
                    print('[!] Cannot add abs diff: mean is missing.')
            if add_z_score:
                if sum([x in ['mean', 'std'] for x in stats]) == 2:
                    mean_name = [col for col in group_agg_df.columns if col.endswith('_MEAN')][0]
                    std_name = [col for col in group_agg_df.columns if col.endswith('_STD')][0]
                    z_name = mean_name.replace('_MEAN', '_Z_SCORE')
                    if sum([x in [mean_name, std_name] for x in df2.columns]) == 2:
                        df2[z_name] = (df2[by_var] - df2[mean_name]) / df2[std_name]
                else:
                    print('[!] Cannot add z score: mean and/or std are missing.')
                
            df2.drop([by_var] + group_vars, axis=1, inplace=True)
        else:
            df2 = group_agg_df
            df2.drop(group_vars, axis=1, inplace=True)
        
        df2 = df2.set_index('SK_ID_CURR')        
        return df2
    
    def add_group_aggregations(self, df, for_groups_n_vars_n_stats):
        """Add group aggregations (parallelized function)
        
        #([group_list], 'var_string', ([stats_list], bool_diff, bool_z_score, bool_abs_diff))
        #(['g1_1', 'g1_2'], 'v1', (['mean'], True, True, False))
        """
        n_processes = min( len(for_groups_n_vars_n_stats), mp.cpu_count() )

        with mp.Pool(processes=n_processes) as pool:
            results = [pool.apply_async(self.group_aggregation,
                                    args=(df,
                                          gvls[0], # group_vars
                                          gvls[1], # by_var
                                          gvls[2][0], # stats_list 
                                          gvls[2][1], # add_diff 
                                          gvls[2][2], # add_z_score 
                                          gvls[2][3],))  # add_abs_diff 
                                          for gvls in for_groups_n_vars_n_stats]
            output = [p.get() for p in results]
        
        for group_df in output:
            df = df.join(group_df, on='SK_ID_CURR')

        return df
            
    def __feature_engineering(self, df_train, df_test):
        """Add features to the data. This method is used to add features using the complete dataset.
        
        Add new manually engineered features as well as aggregates to the data. Train and test data
        are concatenated before feature engineering and partitioned again after the process is over.

        Args:
            df_train: Pandas dataframe corresponding to the train data.
            df_test: Pandas dataframe corresponding to the test data.

        Returns:
            df_train: Pandas dataframe containing the updated train set.
            df_test: Pandas dataframe containing the updated test set.
        """
        df_train['isTrain'] = 1
        df = df_train.append(df_test, sort=False, ignore_index=True)
        ##################################################################        
        df['N_BU_DEBT_TO_INCOME_RATIO'] = df['AG_BUm1_AMT_CREDIT_SUM_DEBT_SUM'] / df['AMT_INCOME_TOTAL']
        df['N_BU_DEBT_ANNUITY_TO_INCOME'] = df['AG_BUm1_AMT_ANNUITY_SUM'] / df['AMT_INCOME_TOTAL']
        
        df['MEAN_EXT_BINS'] = np.digitize(df['N_EXT_SOURCE_MEAN'], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1])
        df['AGE_BINS'] = np.digitize(df['YEARS_BIRTH'], [18, 25, 30, 35, 45, 55, 65, 100])
        df['POP_BINS'] = np.digitize(df['REGION_POPULATION_RELATIVE'], [0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08])
        df['TENURE_BINS'] = np.digitize(df['YEARS_EMPLOYED'], [0, 1, 3, 5, 10, 60])
        df['CREDIT_TO_INCOME_BINS'] = np.digitize(df['N_CREDIT_TO_INCOME_RATIO'], [0, 0.5, 2, 5, 10, 100])
        MEAN_MAX = ['mean', 'max']
        MEAN_STD = ['mean', 'std']
        MEAN_MED = ['mean', 'median']
        MEAN_MED_STD = ['mean', 'median', 'std']
        FOURS = ['mean', 'min', 'max', 'std']
        
        glvs = [
                (['OCCUPATION_TYPE'], 'N_EXT_SOURCE_MEAN', (['mean'], True, False, True)),
                (['AGE_BINS'], 'N_BU_UTIL_ACTIVE_ALL_N_UTIL', (['mean'], True, False, True)),
                (['AGE_BINS', 'NAME_HOUSING_TYPE'], 'N_BU_COUNT_ACTIVE_CREDIT_CARD', (MEAN_STD, True, True, True)),
                (['NAME_CONTRACT_TYPE'], 'N_CREDIT_TERM_YEARS', (MEAN_MED_STD, True, True, True)),
                (['NAME_INCOME_TYPE'], 'N_CREDIT_TERM_YEARS', (MEAN_MED_STD, True, True, True)),
                (['NAME_HOUSING_TYPE'], 'N_CREDIT_TERM_YEARS', (MEAN_MED_STD, True, True, True)),
                (['CREDIT_TO_INCOME_BINS'], 'N_BU_DEBT_TO_INCOME_RATIO', (MEAN_MED_STD, True, True, True)), # should talk about type of credit (e.g. mortgage, car loan)
                (['CREDIT_TO_INCOME_BINS'], 'N_CREDIT_TERM_YEARS', (MEAN_MED_STD, True, True, True)), # should talk about type of credit (e.g. mortgage, car loan)
                (['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE'], 'N_BU_UTIL_ACTIVE_ALL_N_UTIL', (['mean'], True, False, True)),
                (['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE'], 'N_BAD_COMPANY', (MEAN_MED_STD, True, True, True)),
                (['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE'], 'N_EXT_SOURCE_MEAN', (MEAN_MED_STD, True, False, True)),
                (['OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE'], 'N_CREDIT_TERM_YEARS', (MEAN_MED_STD, True, True, True)),
                (['MEAN_EXT_BINS', 'CODE_GENDER'], 'N_BU_DEBT_TO_INCOME_RATIO', (MEAN_MED_STD, True, True, True)),
                (['ORGANIZATION_TYPE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'AMT_INCOME_TOTAL', (MEAN_MED_STD, True, True, True)),
                (['NAME_EDUCATION_TYPE', 'NAME_CONTRACT_TYPE', 'CODE_GENDER'], 'N_LTV_RATIO', (['mean'], True, False, True)),
                (['AGE_BINS', 'NAME_INCOME_TYPE'], 'N_CC_UTIL_TREND7', (MEAN_MED_STD, True, True, False)),
                (['AGE_BINS', 'NAME_CONTRACT_TYPE'], 'N_BU_PMT_RATE', (MEAN_MED_STD, True, True, True)),
                (['AGE_BINS', 'NAME_EDUCATION_TYPE'], 'N_BU_PMT_RATE', (MEAN_STD, True, True, False)),
                (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'N_CREDIT_PER_CAPITA', (MEAN_MED_STD, True, True, False)),
                (['AGE_BINS', 'NAME_HOUSING_TYPE'], 'N_CC_MONTH1_N_UTIL', (MEAN_STD, True, True, True)),
                (['AGE_BINS', 'NAME_EDUCATION_TYPE'], 'N_CC_MONTH1_N_UTIL', (MEAN_STD, True, True, True)),
                (['AGE_BINS', 'OCCUPATION_TYPE'], 'N_CC_MONTH1_N_UTIL', (MEAN_STD, True, True, True)),
                (['CREDIT_TO_INCOME_BINS'], 'N_CC_MONTH1_N_UTIL', (MEAN_MED_STD, True, True, True)),
                (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'AMT_ANNUITY', (['mean'], True, False, True)),
                (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'AMT_CREDIT', (['mean'], True, False, True)),
                (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], 'N_EXT_SOURCE_MEAN', (['mean'], True, False, True)),
                (['CODE_GENDER', 'ORGANIZATION_TYPE'], 'AMT_ANNUITY', (['mean'], True, False, True)),
                (['CODE_GENDER', 'ORGANIZATION_TYPE'], 'AMT_CREDIT', (['mean'], True, False, True)),
                (['CODE_GENDER', 'ORGANIZATION_TYPE'], 'N_EXT_SOURCE_MEAN', (['mean'], True, False, True)),
                (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'AMT_ANNUITY', (['mean'], True, False, True)),
                (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'CNT_CHILDREN', (['mean'], True, False, True)),
                (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], 'YEARS_ID_PUBLISH', (['mean'], True, False, True)),
                (['OCCUPATION_TYPE'], 'AMT_ANNUITY', (['mean'], True, False, True)),
                (['OCCUPATION_TYPE'], 'YEARS_EMPLOYED', (['mean'], True, False, True)),
                (['OCCUPATION_TYPE'], 'N_PMT_RATE', (['mean'], True, False, True)),
                (['OCCUPATION_TYPE'], 'YEARS_ID_PUBLISH', (['mean'], True, False, True)),
                (['TENURE_BINS'], 'N_BU_UTIL_ACTIVE_ALL_N_UTIL', (['mean'], True, False, True)),
                (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'N_EXT_SOURCE_MEAN', (['mean'], True, False, True)),
                (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], 'N_EXT_SOURCE_MEAN', (['mean'], True, False, True))
               ]
        
        for g in tqdm(glvs):
            df_agg = self.group_aggregation(df, g[0], g[1], g[2][0], add_diff=g[2][1], add_z_score=g[2][2], add_abs_diff=g[2][3])
            df = df.join(df_agg, on='SK_ID_CURR')

        df.drop(['MEAN_EXT_BINS', 'POP_BINS', 'AGE_BINS', 'CREDIT_TO_INCOME_BINS', 'TENURE_BINS'],
                axis=1, inplace=True, errors='ignore')
        
        ##################################################################
        df_train = df[df['isTrain']==1].drop(['isTrain'], axis=1)
        df_test = df[df['isTrain'].isnull()].drop(['isTrain'], axis=1)
        return df_train, df_test


# In[5]:


hc = HomeCredit()


# In[6]:


hc.load_all({'application':False, 'previous':False, 'installments':False, 'card':False, 'pos':False, 'bureau':False})


# In[7]:


lgb_model_params = {'n_estimators': 4000, 
            'learning_rate': 0.02,
            'boosting_type': 'goss',
            'num_leaves': 28,
            'feature_fraction': 0.075,
            'subsample': 0.87,
            'max_depth': 8,
            'reg_alpha': 0.0,
            'reg_lambda': 0.031,
            'min_split_gain': 0.025,
            'min_child_weight': 40,
            'silent': -1,
            'verbose':-1,
            'importance_type': 'gain',
            'n_jobs': 1}
hc.run_model_cv(LGBMClassifier,
             model_params=lgb_model_params,
            fit_params={'eval_metric': 'auc', 'verbose': 250, 'early_stopping_rounds': 200}, 
            train_folds=[])

