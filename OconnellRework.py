from BinaryRework import *

import os
import pandas as pd

from typing import Literal, Union, Tuple
Series = pd.Series

class OConnellUtils:
    @staticmethod 
    def _averagecurve_file(target_obj: 'Binary', filepath: str) -> 'Binary':
        filepath = f'{filepath}/ Average.csv'
        
        fourier_series = target_obj.fourier_fit()
        data = target_obj.get_asym(fourier_series).to_pandas()
        data.index = [target_obj.id.lstrip('0')]

        datafile = None 
        if os.path.exists(filepath):
            datafile = pd.read_csv(filepath, index_col = 0)
            datafile.index = datafile.index.astype(str)

        if datafile is None:
            data.to_csv(filepath)
            return target_obj

        datafile = pd.concat([datafile.drop(index=data.index, errors='ignore'), data], axis = 0)
        datafile.to_csv(filepath)

        return target_obj 
        
    @staticmethod
    def _singletarget_file(target: Union[int, str], author: Literal['kepler', 'tess', 'k2'], by: Tuple[int, int] = (1, None), filepath: str = f'{os.getcwd()}/INSERT AUTHOR HERE Data', **kwargs) -> 'Binary':
        target_obj = Binary(target, author, **kwargs) 

        n_cyc = by[0]
        n_points = by[1]


        filepath = f'''{filepath.replace('INSERT AUTHOR HERE', author.capitalize())}/{n_cyc} Cycles/{n_points} Points''' if 'Points' not in filepath else filepath
        datafile = None 
        all_cyc_data = []

        errors = [] 

        os.makedirs(filepath, exist_ok=True)

        print(f'Target {target}: {n_cyc} Cycles with {n_points} Points')
        print(f'Status: 0 / {n_cyc}'.ljust(50), end = '\r')
        for cyc in range(0, n_cyc):
            try:
                fourier_series = target_obj.fourier_fit(start_index = cyc * n_points, n_points = n_points)
                data = target_obj.get_asym(fourier_series).to_pandas()
                all_cyc_data.append(data)
            except BinaryError:
                errors.append(cyc)
                continue 

            print(f'Status: {cyc + 1} / {n_cyc}'.ljust(50), end = '\r')

        if datafile is None:
            datafile = pd.concat(all_cyc_data, axis = 0)
        else:
            datafile = pd.concat(all_cyc_data + [datafile], axis = 0)
        

        print(f'Status: Saving File...'.ljust(50), end = '\r')
        
        datafile.to_csv(f'{filepath}/{target_obj.id}.csv')
        
        print(f'Status: File Saved'.ljust(50), end = '\r')
        print(f'Status: Finished'.ljust(50), end = '\n')    
        print(f'Errored Cycles: {errors}', end='\n\n')

        return target_obj
        
    @staticmethod
    def _multitarget_file(targets: Series, author: Literal['kepler', 'tess', 'k2'], by: Tuple[int, int] = (1, None), filepath: str = f'{os.getcwd()}/INSERT AUTHOR HERE Data', **kwargs) -> list['Binary']:
        filepath = f'{filepath.replace("INSERT AUTHOR HERE", author.capitalize())}/{by[0]} Cycles/{by[1]} Points'
        all_target_obj = [] 
            
        for target in targets:
            target_obj = OConnellUtils._singletarget_file(target, author, by, filepath, **kwargs)
            target_obj = OConnellUtils._averagecurve_file(target_obj, filepath)
            all_target_obj.append(target_obj)
            
        return all_target_obj

        
    @staticmethod 
    def get_file(target: Union[Series, str, int],  author: Literal['kepler', 'tess', 'k2'], by: Tuple[int, int] = (1, None), filepath: str = f'{os.getcwd()}/INSERT AUTHOR HERE Data', **kwargs) -> Union['Binary', list['Binary']]:
        filepath = filepath.replace('INSERT AUTHOR HERE', author.capitalize())
        if isinstance(target, (list, tuple, Series)):
            
            if len(target) > 1: 
                return OConnellUtils._multitarget_file(target, author, by, filepath, **kwargs)
            
            return OConnellUtils._singletarget_file(target[0], author, by, filepath, **kwargs)
        
        return OConnellUtils._singletarget_file(target[0], author, by, filepath, **kwargs)

    @staticmethod 
    def plot(file : pd.Dataframe, attributes : Union[Literal['LCA', 'OER', 'ΔI'], list[Literal['LCA', 'OER', 'ΔI']]) -> None:
        attributes = np.atleast_1d(attributes)
        
        for attribute in attributes:
            plt.scatter(file[attribute].index, file[attribute])
            plt.title(f'{attribute} vs Cycles')
            plt.show()

        return 
        