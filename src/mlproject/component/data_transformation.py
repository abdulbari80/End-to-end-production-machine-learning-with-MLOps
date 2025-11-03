from src.mlproject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from src.mlproject import logging
import pandas as pd
import os
import joblib
from box.exceptions import BoxValueError

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from typing import Literal

# Import Statistics libraries
from scipy import stats
from scipy.stats import norm

# Import Scikit-learn for Machine Learning libraries
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
#from feature_engine.encoding import RareLabelEncoder

# Import country code libraries
import pycountry

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def _regroup_job_titles(self, job_title):
        """This regroup over 300 job titles into similar categories"""
        try:
            bi_analyst = [
                'BI Data Analyst', 'Business Data Analyst', 'BI Developer', 'BI Analyst', 
                'Business Intelligence Engineer', 'BI Data Engineer',
                'Business Intelligence Analyst','Power BI Developer', 
                'Business Intelligence Developer', 'Business Intelligence', 'BI Engineer']
            
            data_analyst = [
                'Analyst','Data Analyst', 'Data Quality Analyst', 'Product Data Analyst', 
                'Data Analytics Lead', 'Data Lead', 'Finance Data Analyst', 'Insight Analyst', 
                'Lead Data Analyst', 'Financial Data Analyst', 'Staff Data Analyst', 
                'Compliance Data Analyst', 'Data Analytics Engineer', 'Data Operations Analyst', 
                'Data Analytics Lead', 'Data Analytics Specialist', 'Data Analytics Consultant', 
                'Marketing Data Analyst', 'Principal Data Analyst', 'Data Management Analyst', 
                'Quantitative Analyst', 'Data Reporting Analyst']
            data_scientist = [
                'Data Scientist', 'Applied Scientist', 'Research Scientist', 'Lead Data Scientist',
                '3D Computer Vision Researcher', 'Deep Learning Researcher', 'Staff Data Scientist', 
                'Data Science Lead', 'Data Science Consultant', 'Product Data Scientist', 
                'Data Science Tech Lead','Applied Data Scientist', 'Principal Data Scientist', 
                'Data Science Engineer', 'Data Modeler', 'Decision Scientist']
            
            ai_engineer = [
                'AI/Computer Vision Engineer', 'Computer Vision Software Engineer', 'AI Scientist', 
                'AI Programmer', 'AI Developer', 'Computer Vision Engineer', 'AI Architect', 
                'Deep Learning Engineer', 'AI Specialist']
            
            ml_engineer = [
                'Machine Learning Engineer', 'ML Engineer', 'Lead Machine Learning Engineer',
                'Principal Machine Learning Engineer', 'Machine Learning Scientist',
                'MLOps Engineer', 'NLP Engineer','Applied Machine Learning Scientist', 
                'Machine Learning Software Engineer', 'Applied Machine Learning Engineer', 
                'Machine Learning Developer', 'Machine Learning Infrastructure Engineer']
            
            data_engineer = [
                'Data Engineer', 'ETL Developer', 'Big Data Engineer', 'Azure Data Engineer',
                'Lead Data Engineer', 'Analytics Engineer', 'Data Operations Engineer',
                'Cloud Data Engineer', 'Marketing Data Engineer', 'ETL Engineer',
                'Principal Data Engineer', 'Software Data Engineer', 'Software Data Engineer',
                'Cloud Database Engineer', 'Data DevOps Engineer', 'Data Architect', 'Data Integration Engineer',
                'Big Data Architect', 'Data Infrastructure Engineer', 'Cloud Data Architect', 
                'Cloud Data Architect', 'Principal Data Architect', 'Architect', 'Data Developer']

            executive = [
                'Head of Data', 'Data Science Manager', 'Director of Data Science', 'Manager',
                'Head of Data Science', 'Data Scientist Lead', 'Head of Machine Learning', 
                'Manager Data Management', 'Data Analytics Manager', 'Data Manager', 
                'Data Specialist', 'Data Management Specialist', 'Engineering Manager', 
                'Data Lead', 'Data Strategist', 'Machine Learning Manager', 'Technical Lead', 
                'Analytics Lead', 'Data Governance Lead', 'Lead Engineer']
            
            ai_ml_researcher = [
                'Machine Learning Researcher', 'Machine Learning Research Engineer',
                'Research Engineer', 'Research Analyst', 'AI Researcher']
            
            soft_engineer = [
                'Software Engineer', 'Software Development Engineer' , 'Developer', 
                'Software Developer', 'DevOps Engineer', 'Solution Architect', 'Backend Engineer',
                'Solutions Engineer', 'Full Stack Engineer', 'Solutions Architect',
                'Full Stack Developer']

            it_engineer = [
                'Engineer', 'Systems Engineer', 'Platform Engineer', 'Site Reliability Engineer',
                'Cloud Engineer'],
            
            other_data_prof = [
                'Data Product Manager', 'Data Governance', 'Data Governance Analyst', 'Data Product Owner']
            
            if job_title in data_scientist:
                return "Data Scientist"
            elif job_title in bi_analyst:
                return "BI Analyst"
            elif job_title in ml_engineer:
                return "ML Engineer"
            elif job_title in data_engineer:
                return "Data Engineer"
            elif job_title in executive:
                return "AI/ DS Executive"
            elif job_title in data_analyst:
                return "Data Analyst"
            elif job_title in ai_engineer:
                return "AI Engineer"
            elif job_title in ai_ml_researcher:
                return "AI/ ML Researcher"
            elif job_title in soft_engineer:
                return "Software Engineer"
            elif job_title in it_engineer:
                return "IT Engineer"
            elif job_title in other_data_prof:
                return "Other Data Profession"
            else:
                return job_title
        except BoxValueError as e:
            logging.error(f"Error: {e}")

    def _feature_engineer(self, df: pd.DataFrame, 
                        old_feature: str = None,
                        new_feature: str = None,
                        option: Literal['top_n', 'threshold'] = 'top_n',
                        top_n: int = 5,
                        threshold: int =100):
        try: 
            if option == 'top_n':
                top_cat_index = df[old_feature].value_counts().nlargest(top_n).index
            elif option == 'threshold':
                freq_arr = df[old_feature].value_counts()
                top_cat_index = freq_arr[freq_arr >= threshold].index
            else:
                raise ValueError(f"Didn't choose an option from ['top_n', 'threshold']")
            
            df[new_feature] = df[old_feature].apply(
                lambda x: 'Other' if x not in top_cat_index else x)
            return df
        except BoxValueError as e:
            logging.error("Error: {e}")
            
    def _country_code_to_name(self, code):
        """This converts ISO 3166 country code to country name"""
        try:
            return pycountry.countries.get(alpha_2=code).name
        except:
            return None  # Use None so it can be safely skipped later if needed
        
    def _adjust_salary(self, data:pd.DataFrame):
        try:
            year = data['work_year']
            country = data['company_location']
            salary_usd = data['salary_in_usd']
            # Inflation rates
            us_inflation_rates = {2019: 0.018, 2020: 0.012, 2021: 0.047, 
                                2022: 0.08, 2023: 0.041, 2024: 0.029, 2025: 0.02}
            global_inflation_rates = {2019: 0.019, 2020: 0.019, 2021: 0.035, 
                                    2022: 0.057, 2023: 0.049, 2024: 0.058, 2025: 0.036}
            adjsuted_salary = salary_usd
            if country == 'United States':
                inflation_rates = us_inflation_rates
            else:
                inflation_rates = global_inflation_rates
            for yr in range(year, 2025):
                inflation_rate = inflation_rates[yr]
                adjsuted_salary *= (1+inflation_rate)
            # adjsuted_salary = int(round(adjsuted_salary, 0))

            return adjsuted_salary
        except BoxValueError as e:
            logging.error(f"Error: {e}")

    def _get_data_processing(self):
        """This processes raw data including feature engineering as well as
        stores and returns a clean dataframe.  
        """
        try: 
            df = pd.read_csv(self.config.data_path)
            #df = df[df['work_year'].isin([2025])]
            logging.info("Data loaded from data_ingestion directory")
            # Remove duplicate instances
            df.drop_duplicates(inplace=True)
            # abbreviated exp levels are replaced with full form
            df['experience_level'] = df['experience_level'].replace(
                {'SE': 'Senior',
                'EN': 'Entry level',
                'EX': 'Executive level',
                'MI': 'Mid/ Intermediate level'})
            # abbreviated employment types are replaced with full name
            df['employment_type'] = df['employment_type'].replace(
                {
                'FL': 'Freelancer',
                'CT': 'Contractor',
                'FT' : 'Full-time',
                'PT' : 'Part-time'})
            # remote work ratio is converted from numeric to categorical
            df['remote_ratio'] = df['remote_ratio'].astype(str)
            df['remote_ratio'] = df['remote_ratio'].replace(
                {'0': 'On-site',
                '50': 'Hybrid',
                '100': 'Remote'})
            # Company size is mapped to full form
            df['company_size'] = df['company_size'].replace(
                {'L': 'Large',
                'M': 'Medium',
                'S': 'Small'})

            # create new column with regrouped job categories 
            df['job_title_regroup'] = df['job_title'].apply(self._regroup_job_titles)
            # Job title with frequency <1000 categorised as 'Other'
            df = self._feature_engineer(df, 
                                        old_feature='job_title_regroup',
                                        new_feature='job_title_freq',
                                        option='top_n',
                                        top_n=8)
            # drop irrelevant job titles
            df = df[~df['job_title_regroup'].isin(['Engineer', 'Other'])]
            # Country codes are mapped to country names
            df['company_location'] = df['company_location'].apply(
                self._country_code_to_name)
            df['employee_residence'] = df['employee_residence'].apply(
                self._country_code_to_name)
            # countries beyond top 7 are re-labelled as 'other'
            df = self._feature_engineer(df, old_feature='employee_residence',
                                        new_feature='employee_residence_top',
                                        option='top_n',
                                        top_n=7)
            # Select top 7 categories in company_location and remaining in other category
            df = self._feature_engineer(df, old_feature='company_location',
                                        new_feature='company_location_top', 
                                        option='top_n',
                                        top_n=7)
            
            df['inflation_adj_salary'] = df.apply(self._adjust_salary, axis=1)
            return df
        except BoxValueError as e:
            logging.error(f"Error: {e}")
    
    def _remove_salary_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            q1 = df['inflation_adj_salary'].quantile(0.25)
            q3 = df['inflation_adj_salary'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
                
            return df[(df['inflation_adj_salary'] >= lower_bound) & 
                    (df['inflation_adj_salary'] <= upper_bound)]
        except BoxValueError as e:
            logging.error("Error: {e}")

    def _select_columns(self):
        """This drops redundant columns"""
        try:
            df = self._get_data_processing()
            df = self._remove_salary_outliers(df=df)
            # Select necessary columns
            cols_df = df.columns.to_list()
            final_cols_to_select = []
            cols_to_keep = ['experience_level', 'employment_type', 'remote_ratio',
                            'company_size', 'job_title_freq', 'employee_residence_top',
                            'company_location_top', 'inflation_adj_salary']
            for col in cols_to_keep:
                if col in cols_df:
                        final_cols_to_select.append(col)
            df_sub = df[final_cols_to_select]
            return df_sub
    
        except BoxValueError as e:
            logging.error(f"Error: {e}")
    
    
    def _get_col_transformer_pipeline(self) -> ColumnTransformer:
        """This builds a pipeline to process data"""
        logging.info('Data transformation starts >>>>>')
        try:
            cat_columns = ['experience_level', 
                           'employment_type',
                           'remote_ratio', 
                           'company_size', 
                           'job_title_freq', 
                           'employee_residence_top', 
                           'company_location_top']
            
            cat_pipeline = Pipeline(
                steps = [
                    ('cat_pipeline', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encode', OneHotEncoder(handle_unknown='ignore',
                                                     sparse_output=False))
                        ])

            trans_cols = ColumnTransformer(
                [
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ],
                remainder='drop'
            )
            logging.info('Data preprocessing pipeline created.')
            return trans_cols           
        except BoxValueError as e:
            logging.error(f"Error: {e}")

    def initiate_data_transformation(self, test_size:float=0.20):
        """ This stores data transormation pipeline object 
        with train and test arrays in the artifacts.
        """
        try:
            df = self._select_columns()
            df_train, df_test = train_test_split(df,
                                       test_size=0.25,
                                       random_state=42)
            df_train, df_val = train_test_split(df_train, test_size=test_size, 
                                                random_state=42)
            df_train = pd.concat([df_train, df_test], axis=0)
            dataset = [(df_train, 'train.csv'),
                       (df_val, 'test.csv')]
            # Save train and test datasets in the artifacts
            for data in dataset:
                data[0].to_csv(os.path.join(self.config.root_dir, data[1]), 
                               index=False)
            logging.info("dataset split into train and test sets and saved to artifacts")
            if 'inflation_adj_salary' in df.columns.to_list():
                target_variable = 'inflation_adj_salary'
            else:
                  logging.info("Sorry, 'inflation_adj_salary' not available")
            X_train = df_train.drop(columns=[target_variable], axis=1)
            y_train = df_train[target_variable]
            X_test = df_test.drop(columns=[target_variable], axis=1)
            y_test = df_test[target_variable]
            # Instantiate column transformer class
            column_transormer_pipeline = self._get_col_transformer_pipeline()
            X_train_arr = column_transormer_pipeline.fit_transform(X_train)
            X_test_arr = column_transormer_pipeline.transform(X_test)
            logging.info("Data columns are transformed")
            # save data transformation pipeline object to artifacts
            joblib.dump(column_transormer_pipeline, 
                        os.path.join(self.config.root_dir, 
                                     self.config.data_transform_obj_name))
            logging.info(f"Custom column transformer saved to {self.config.root_dir}")
            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]
            joblib.dump(train_arr, os.path.join(self.config.root_dir, 
                                                self.config.train_array))
            joblib.dump(test_arr, os.path.join(self.config.root_dir, 
                                               self.config.test_array))
            
        except BoxValueError as e:
            logging.error(f"{e}")