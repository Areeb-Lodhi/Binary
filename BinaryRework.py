from lightkurve import search_targetpixelfile
from lightkurve import search_lightcurve

import pandas as pd
from pandas import Series
import numpy as np

import matplotlib
plt = matplotlib.pyplot
FuncAnimation = matplotlib.animation.FuncAnimation
from IPython.display import HTML, display
import ipywidgets as widgets

from scipy.optimize import curve_fit, minimize_scalar
from scipy.integrate import quad

import sys
import os
import warnings

import ast

from typing import Union, Callable, Literal, Tuple, Final 
from numbers import Number
from dataclasses import dataclass

import importlib.util
import subprocess

def install_if_missing(package_name, import_name=None):
    import_name = import_name or package_name
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])



#---------------------------------------------------------------------------------------------------------------------------------------------------------------

class FourierSeries:
    @staticmethod
    def get_series(coeffs: Series) -> Callable[ Union[Series, Number], Union[np.ndarray, float] ]:
        """
        Constructs a callable Fourier series function with respect to x.
    
        Parameters
        ----------
        coeffs : A series of coefficiants
            A series of Fourier coefficients formatted as [a0, a1, b1, a2, b2, ...]
    
        Returns
        -------
        Callable[[Series or Number], ndarray or float]
            A function representing the Fourier series mathematically. It accepts either:
            - A scalar `x` (returns a float)
            - A Series of `x` values (returns a NumPy ndarray of dtype float)
        """
        a0 = coeffs[0] 
        an = np.array(coeffs[1::2]) 
        bn = np.array(coeffs[2::2]) 
        omega = 2 * np.pi * np.arange(1, an.size + 1) 
    
        def fourier_series(x): 
            x = np.atleast_1d(x)
            result = a0/2 + np.sum(an[:, None] * np.cos(omega[:, None] * x) + bn[:, None] * np.sin(omega[:, None] * x), axis = 0) 
            return result.item() if np.isscalar(result) or result.size == 1 else result

        return fourier_series 

    @staticmethod
    def get_extrema(f : Callable[Number, Number], bounds: Tuple[Number, Number], maximum: bool = False) -> Tuple[Number, Number]:
        """
        Finds specifed extrema of a Callable function using the bounded method. 

        Parameters
        ----------
        f : Callable[Number, Number]
            Function attempting to minimize 
        bounds : (Number, Number)
            Restricts values from bounds[0] to bounds[1]
        maximum : bool
            Indicates whether the extram attempting to find is a maximum or not (default is False)

        Returns
        -------
        (Number, Number)
            A pair of numbers reprsesenting extema value and extrema location respectively 
        """
        sign = -1 if maximum else 1
        f_objective = lambda x : sign * f(x)
        minimized_series_result = minimize_scalar(f_objective, bounds = bounds, method='bounded')
        return (sign * minimized_series_result.fun, minimized_series_result.x)
    
    def __init__(self, coeffs_and_cov: Series) -> None:
        """
        Initialize the FourierSeries.

        Parameters
        ----------
        coeffs_and_cov : Series of Series
            A series containing two elements:
                - Index 0: A series of Fourier coefficients formatted as [a0, a1, b1, a2, b2, ...]
                - Index 1: A 2D series of covariance terms of Fourier coefficients 
        """
        self.coeffs: Final = coeffs_and_cov[0]
        self.coeffs_cov: Final = coeffs_and_cov[1]
        self.coeffs_err: Final = np.sqrt(np.diag( self.coeffs_cov ))
        
        if len(self.coeffs) <= 1 or len(self.coeffs) % 2 == 0:
            return ValueError('Coefficient list must contain an odd number of elements: [a0, a1, b1, a2, b2, ...]')
        
        self.series_func = FourierSeries.get_series(self.coeffs) 

        self.min_loc = None
        self.max_loc = None
        self.min_val = None 
        self.max_val = None 

    def __call__(self, x: Union[Number, Series]) -> Callable[ Union[Series, Number], Union[np.ndarray, float] ]:
        """
        Evaluate the function at value(s) of 'x'

        Parameters
        ----------
        x : Number or a Series of Numbers
            Value or series of values to evaluate the Fourier series

        Returns
        -------
        float or np.ndarray
            The evaluated result of the Fourier series:
                - A single number if 'x' is scalar
                - A np.ndarray of floats if 'x' is a series
        """
        return self.series_func(x)

    def __str__(self) -> str:
        terms = [f'{coeffs[0]/2:0.3e}']

        for i in range(0, len(self.coeffs[1:]) // 2):
            terms.append(f'{self.coeffs[2*i+1]:0.3e}cos({2*(i+1)}πφ)')
            terms.append(f'{self.coeffs[2*i+2]:0.3e}sin({2*(i+1)}πφ)')

        return ' + '.join(terms)

    def __repr__(self) -> str:
        return f'<FourierSeries: n_terms={len(self.coeffs)}; min={self.min_val}, min_loc={self.min_loc}; max={self.max_val}, max_loc={self.max_loc}>'
        

    def get_min(self) -> float:
        """
        Gets the minumum value of the Fourier series and initializes self.min_val with said value.

        Returns
        -------
        float 
            Minimum value of the Fourier series
        """
        if self.min_loc is not None and self.min_val is not None:
            return self.min_val

        self.min_val, self.min_loc = FourierSeries.get_extrema(self.series_func, bounds=(0,1), maximum=False)

        return self.min_val

    def get_max(self) -> float:
        """
        Gets the maximum value of the Fourier series and initializes self.max_val with said value.

        Returns
        -------
        float 
            Maximum value of the Fourier series
        """
        if self.max_loc is not None and self.max_val is not None:
            return self.max_val
        
        self.max_val, self.max_loc = FourierSeries.get_extrema(self.series_func, bounds=(0,1), maximum=True)

        return self.max_val

    def get_min_loc(self) -> float:
        """
        Gets the minumum location of the Fourier series and initializes self.min_loc with said value.

        Returns
        -------
        float 
            Minimum location of the Fourier series
        """
        _ = self.get_min()
        return self.min_loc
    
    def get_max_loc(self) -> float:
        """
        Gets the maximum location of the Fourier series and initializes self.max_loc with said value.

        Returns
        -------
        float 
            Maximum location of the Fourier series
        """
        _ = self.get_max()
        return self.max_loc

    def plot(self, ax: matplotlib.axes.Axes=None, **kwargs) -> matplotlib.axes.Axes:
        """
        Plots the Fourier series on specified bounds. 

        Parameters 
        ----------
        ax : matplotlib.axes.Axes 
            An axis object to plot on, will create a new one if None (default is None)

        Returns
        -------
        matplotlib.axes.Axes
            An axis object with the Fourier series plotted on it 
        """
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault('bounds', (0, 1))
        kwargs.setdefault('n_points', 1000)
        
        x = np.linspace(kwargs['bounds'][0], kwargs['bounds'][1], kwargs['n_points'])
        y = self.series_func(x)

        kwargs.pop('bounds')
        kwargs.pop('n_points')

        ax.plot(x, y, label='Fourier Fit', **kwargs)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')
        ax.legend()
    
        return ax

    def deriv(self, n: int = 1) -> 'FourierSeries':
        """
        Calculates the n-th order derivative of the current FourierSeries object and intializes a new FourierSeries object with the new coefficients.

        Parameters
        ----------
        n : int 
            The order of the derivative.

        Returns
        -------
        FourierSeries Object 
            The FourierSeries object initialized with the new coeffcients after taking the derivatives
        """
        if n<1:
            raise ValueError('Derivative must have an order >= 1')

        an = self.coeffs[1::2]
        bn = self.coeffs[2::2]

        deriv_factor = (2 * np.pi * np.arange(1, len(an) + 1)) ** n

        an *= deriv_factor * [1, 1, -1, -1][(n + 1) % 4]
        bn *= deriv_factor * [1, 1, -1, -1][n % 4]

        if n % 2 == 1:
            an, bn = bn, an

        coeffs = []
        for a, b in zip(an, bn):
            coeffs.append([a, b])

        cov_placeholder = np.zeros((len(coeffs), len(coeffs)))

        return FourierSeries((coeffs, cov_placeholder))


#-----------------------------------------------------------------------------------------------------------------------------------

@dataclass
class AsymmetryData:
    """
    Stores asymmetry metrics for any Fourier series (but mainly binary light curves)

    Attributes
    ----------
    LCA : float 
        Light Curve Asymmetry
    OER : float
        OConnell Effect Ratio
    I : (float, float) 
        Primary maximum and secondary maximum respectively 
    LCA_err : float 
        Error in Light Curve Asymmetry 
    OER_err : float
        Error in OConnell Effect Ratio
    I_err : float
        Error in the Fourier fit 
    """
    LCA: float = None
    OER: float = None
    I: Tuple[float, float] = None 
    
    LCA_err: float = None 
    OER_err: float = None 
    I_err: float = None 

    @staticmethod 
    def _LCA_err(fourier_series: 'FourierSeries') -> float:
        LCA = 1
        LCA_grad_vector = [0]

        for i, coeff in enumerate(fourier_series.coeffs[1:]):
            if i % 2 == 0:
                N = lambda x: 0
                D = lambda x: 0
            
        
        return 0 

    @staticmethod 
    def _OER_err(fourier_series: 'FourierSeries') -> float:
        return 0 

    @staticmethod 
    def _I_err(fourier_series: 'FourierSeries') -> Callable[ Union[Series, Number], Union[Series, Number] ]:
        a0 = fourier_series.coeffs_err[0]
        an = fourier_series.coeffs_err[1::2]
        bn = fourier_series.coeffs_err[2::2]

        omega = 2*np.pi*np.arange(1, len(an) + 1)

        def err_function(x):
            x = np.atleast_1d(x)
            
            cos_terms = an[:, None] *( np.cos(omega[:, None] * x) ** 2 )
            sin_terms = bn[:, None] * ( np.sin(omega[:, None] * x) ** 2 )

            result = a0**2/4 + np.sum(cos_terms + sin_terms)
            result = result ** (0.5) 

            return result.item() if np.isscalar(result) or result.size == 1 else result

        return err_function
            
    
    @staticmethod
    def calculate_LCA(fourier_series: 'FourierSeries') -> Tuple[float, float]:
        with warnings.catch_warnings(record=True) as w:
            LCA_numerator = (lambda x : (fourier_series(x) - fourier_series(1-x))) # **(1/2)
            # LCA_denominator = lambda x: fourier_series(x) - fourier_series(0)
            
            LCA = quad(LCA_numerator, 0, 0.5, limit = 100)[0] #** (0.5) / quad(LCA_denominator, 0, 0.5, limit = 100)[0]
    
            if w:
                raise BinaryError(ValueError('Error with integration'))
                
        return ( abs(LCA), AsymmetryData._LCA_err(fourier_series) )

    @staticmethod
    def calculate_OER(fourier_series: 'FourierSeries') -> Tuple[float, float]:
        with warnings.catch_warnings(record=True) as w:
            numer = quad(fourier_series, 0, 0.5, limit = 100)
            denom = quad(fourier_series, 0.5, 1, limit = 100)
            f0 = fourier_series(0)
    
            if w:
                raise BinaryError(ValueError('Error with integration'))

            oer = (numer[0] - f0*0.5)/(denom[0] - f0*0.5)
                
        return (oer, AsymmetryData._OER_err(fourier_series))

    @staticmethod
    def calculate_I(fourier_series: 'FourierSeries') -> Tuple[Tuple[float, float], Tuple[float, float]]:
        Ip = 10 * quad(fourier_series, 0.2, 0.3, limit=100)[0]
        Is = 10 * quad(fourier_series, 0.7, 0.8, limit=100)[0]
        
        if np.isnan(Ip - Is):
            raise BinaryError(ValueError('Error with detecting one of the eclipses'))

        err_func = AsymmetryData._I_err(fourier_series)
            
        return ( (Ip, Is), (err_func(0.25), err_func(0.75)) )

    @classmethod
    def calculate(cls, fourier_series: 'FourierSeries') -> 'AsymmetryData':
        """
        Calculates all values of the AsymmetryData class and returns an instance with correctly assigned attributes. 

        Parameters
        ----------
        fourier_series : FourierSeries
            A FourierSeries object of the Fourier series the data is for 

        Returns
        -------
        AsymmetryData Object 
            An object with evaluated attributes for the given Fourier series 
        """
        lca, lca_err = cls.calculate_LCA(fourier_series)
        oer, oer_err = cls.calculate_OER(fourier_series)
        i, i_err = cls.calculate_I(fourier_series)
        return cls(
            LCA=lca,
            OER=oer,
            I=i,
            LCA_err=lca_err,
            OER_err=oer_err,
            I_err=i_err,
        )

    @staticmethod
    def animate(target_obj: 'Binary', by: Tuple[float, float], start_cyc: int = 0, **kwargs):
        kwargs.setdefault('Light Curve', True)
        kwargs.setdefault('Asymmetry', True)
        kwargs.setdefault('LCA', False)
        kwargs.setdefault('OER', False)
        kwargs.setdefault('ΔI', False)
        kwargs.setdefault('Assosiated Errors', False)
        kwargs.setdefault('Morph', False)
        kwargs.setdefault('Classification', False)

        asym = kwargs.get('Asymmetry')
        lc = kwargs.get('Light Curve')
        
        n_cyc = by[0]
        n_points = by[1]

        asym_x = np.linspace(0, 0.5, 500)
        _x = np.linspace(0, 1, n_points)
        fit_x = np.linspace(0, 1, 1000)

        if asym:
            fig, axs = plt.subplots(1, 2 if lc else 1, figsize=(8 * (2 if lc else 1), 4), squeeze=False, constrained_layout=False)
            axs = axs.flatten()
            asym_line = axs[0].scatter(asym_x, asym_x, c=asym_x, cmap='bwr')
            
            if lc:
                data_line, = axs[1].plot(_x, _x, marker='o', markersize='1', linestyle='None', label='Light Curve Data')
                fit_line, = axs[1].plot(fit_x, fit_x, linestyle='-', linewidth='.5', color='orange', label='Fourier Fit')
                
                axs[1].set_xlim(0, 1)
                axs[1].set_title('Light Curve') 

            axs[0].set_xlim(0, 0.5)
            axs[0].set_title('Difference in Flux')
    
        else:
            fig, axs = plt.subplots(1, 1, figsize=(8, 4), squeeze=False)
            axs = axs.flatten()
            data_line, = axs[0].plot(_x, _x, marker='o', markersize='1', linestyle='None', label='Light Curve Data')
            fit_line, = axs[0].plot(fit_x, fit_x, linestyle='-', linewidth='.5', color='orange', label='Fourier Fit')

            axs[0].set_xlim(0, 1)
            axs[0].set_title('Light Curve') 
        
        fig.subplots_adjust(right=0.7) 
        annotation = axs[1 if lc and asym else 0].annotate('', xy=(1, 0), xytext=(10,0), xycoords='axes fraction', textcoords='offset points', fontsize=10, clip_on=False)

        asym_data = [] 
        light_data = []
        fit_data = []
        annotation_text = []
        error = kwargs.get('Assosiated Errors')
        
        for cyc in range(n_cyc):
            fourier_series = target_obj.fourier_fit(start_index = (start_cyc + cyc) * n_points, n_points = n_points)
            if asym:
                asym_data.append(fourier_series(asym_x) - fourier_series(1-asym_x))
                
            if lc:
                fit_data.append(fourier_series(fit_x))
                light_data.append(target_obj.active_data)
            
            total_annotation = [f'Period: {target_obj.period}']
            if kwargs.get('Morph'):
                total_annotation.append(f'Morph: {target_obj.morph}')

            if kwargs.get('Classification'):
                total_annotation.append(f'Classification: {target_obj.classification}')
                
            total_annotation.append('')
            
            if kwargs.get('LCA'):
                lca = AsymmetryData.calculate_LCA(fourier_series)
                total_annotation.append(f'LCA: {lca[0]:.3e}')
                if error:
                    total_annotation.append(f'δLCA: {lca[1]:.3e}')
                    
            if kwargs.get('OER'):
                oer = AsymmetryData.calculate_OER(fourier_series)
                total_annotation.append(f'OER: {oer[0]:.3e}')
                if error:
                    total_annotation.append(f'δOER: {oer[1]:.3e}')
                    
            if kwargs.get('ΔI'):
                I = AsymmetryData.calculate_I(fourier_series)
                total_annotation.append(f'ΔI: {(I[0][0] - I[0][1]):.3e}')
                if error:
                    total_annotation.append(f'δΔI: {I[1]}')

            annotation_text.append('\n'.join(total_annotation))

            
        if asym:
            ymin = min(arr.min() for arr in asym_data)
            ymax = max(arr.max() for arr in asym_data)
            buffer = (ymax - ymin) * 0.10
            axs[0].set_ylim(ymin - buffer, ymax + buffer)
            asym_line.set_clim(min(ymin, -ymax), max(-ymin, ymax))

        if lc:
            ymin = target_obj.pandas_data['flux'].min()
            ymax = target_obj.pandas_data['flux'].max()
            buffer = (ymax - ymin) * 0.10
            axs[1 if asym else 0].set_ylim(ymin - buffer, ymax + buffer)
            
        save = widgets.Button(
            description='Save Animation',
            disabled=False,
            button_style='success',
            icon=''
        )

        annotation_description = f'{target_obj.author.capitalize()} Mission : {target_obj.id}\n'
        
        def update(frame):
            return_val = ()
            annotation.set_text(f'{annotation_description}Cycle {frame+1}\n\n{annotation_text[frame]}')
            
            if asym:
                asym_line.set_offsets(np.c_[asym_x, asym_data[frame]])
                asym_line.set_array(asym_data[frame])
                return_val += (asym_line,)

            if lc:
                fit_line.set_ydata(fit_data[frame])
                data_line.set_xdata(light_data[frame]['phase'])
                data_line.set_ydata(light_data[frame]['flux'])
                return_val += (data_line, fit_line,)
            
            return return_val
            
        ani = FuncAnimation(fig, update, frames=n_cyc, interval=700, blit=True)

        def save_animation(_):
            install_if_missing('pillow')
            ani.save(f'{target_obj.id}_LC_Animation.gif', writer='pillow', fps=2)

        save.on_click(save_animation)
        
        display(HTML(ani.to_jshtml()))
        display(save)
    
        return None

    def to_pandas(self, **kwargs) -> pd.DataFrame:
        kwargs['columns'] = ['Ip', 'Is', 'ΔI', 'ƍΔI', 'OER', 'ƍOER', 'LCA', 'ƍLCA']
        dict_info = { 
            'Ip' : [self.I[0]], 
            'Is' : [self.I[1]], 
            'ΔI' : [self.I[0] - self.I[1]], 
            'ƍΔI' : [np.sqrt(self.I_err[0] ** 2 + self.I_err[1] ** 2)], 
            'OER' : [self.OER], 
            'ƍOER' : [self.OER_err], 
            'LCA' : [self.LCA], 
            'ƍLCA' : [self.LCA_err]
        }
                      
        return pd.DataFrame(dict_info, **kwargs)

#-----------------------------------------------------------------------------------------------------------------------------------
class BinaryError(Exception):
    pass
#-----------------------------------------------------------------------------------------------------------------------------------
class Binary:
    def __init__(self, target_id: Union[int, str], author: Literal['kepler', 'tess', 'k2'], harm: int=10, cadence: Literal['long', 'short']='long', quarters: Union[list, None]=None, filepath: str=os.getcwd()) -> None:
        self.id = str(target_id).zfill(8)
        
        self.author = author 
        self.cadence = cadence
        self.quarters = quarters

        self.n_harmonics = harm 
        self.flags = []
        
        self.filepath = filepath 
        
        try:
            if author == 'tess':
                self.villinova = None
            elif author == 'k2':
                self.vilinnova = None 
            else:
                self.villinova = pd.read_html(f'https://keplerebs.villanova.edu/overview/?k={self.id.lstrip("0")}')
                self.villinova = pd.concat([self.villinova[0], self.villinova[1]], axis=1)
        except Exception as e:
            raise ValueError(f'No valid id for mission: {self.id}, {author}')

        self.period = float(self.villinova.loc[0, 'period'])
        self.morph = float(self.villinova.loc[0, 'morph'])
        
        self.lightkurve_data = search_lightcurve(self.id, author=self.author, cadence=self.cadence, quarter=quarters).download_all().stitch()
        self.pandas_data = self.lightkurve_data.normalize().to_pandas().dropna(subset=['flux']).sort_values(by='flux')

        temp =  self.lightkurve_data.fold(period=self.period, normalize_phase=True).to_pandas().dropna(subset=['flux']).sort_values(by='flux')

        self.pandas_data['phase'] = temp.index 
        self.pandas_data = self.pandas_data.sort_index()

        self.pandas_data = self.pandas_data.sort_index()

        self.active_data = None 
        self.asym = AsymmetryData()

        self.phased_to_min = False
        self.classification = None

    def classify(self, fourier_coefficients: Series) -> str:
        if len(fourier_coefficients) < 8: # Ensures that their are enough coefficients to use the classification method 
            raise ValueError('Expected a series of minimum length 8.')

        # Defines variables used for checking classification 
        a1 = fourier_coefficients[1]
        a2 = fourier_coefficients[3]
        a4 = fourier_coefficients[7]

        
        # Defines typing
        wu = a4 > a2 * (0.125 - a2) 
        al = ~wu & (np.abs(a1) < 0.05) 
        bl = ~al & ~wu
    
        # Apply classifications
        self.classification = 'Al' if al else self.classification
        self.classification = 'WU' if wu else self.classification
        self.classification = 'BL' if bl else self.classification

        return self.classification

    def fourier_fit(self, start_index: int = 0, n_points: int = None) -> Union['FourierSeries', None]:
        if n_points is None:
            n_points = self.pandas_data.shape[0] - start_index
        
        if start_index + n_points > self.pandas_data.shape[0]:
            return None 

        self.active_data = self.pandas_data.iloc[start_index:start_index + n_points].copy()

        def objective_function(x, *coeffs):
            return FourierSeries.get_series(coeffs)(x)

        initial_guess = [1.0] + [1.0] * (2 * self.n_harmonics)
        coeffs, coeffs_cov = curve_fit( objective_function, self.active_data['phase'], self.active_data['flux'], p0=initial_guess ) 
        fourier_series = FourierSeries( (coeffs, coeffs_cov) )

        if not self.phased_to_min:
            min_loc = fourier_series.get_min_loc()
            
            self.pandas_data['phase'] = (self.pandas_data['phase'] - min_loc) % 1
            self.active_data['phase'] = (self.active_data['phase'] - min_loc) % 1

            coeffs, coeffs_cov = curve_fit( objective_function, self.active_data['phase'], self.active_data['flux'], p0=initial_guess ) 
            fourier_series = FourierSeries( (coeffs, coeffs_cov) )
            
            self.phased_to_min = True 

        self.classify(coeffs)
        
        return fourier_series

    def get_asym(self, fourier_series: 'FourierSeries') -> 'AsymmetryData':
        self.asym = AsymmetryData.calculate(fourier_series)
        return self.asym
        
        

        

    