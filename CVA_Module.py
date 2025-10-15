import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import floor, ceil, log2
from time import perf_counter
from numba import njit
from scipy.stats import qmc, norm

class CVA():
    def __init__(self, CHART_DIMENTIONS_SINGLE_CHART:tuple=(20,7), CHART_DIMENTIONS_DOUBLE_CHART:tuple=(20,11)):
        self.CHART_DIMENTIONS_SINGLE_CHART = CHART_DIMENTIONS_SINGLE_CHART
        self.CHART_DIMENTIONS_DOUBLE_CHART = CHART_DIMENTIONS_DOUBLE_CHART
        self.df_cds = pd.read_csv('data/same_day_cds_london_cds_generic_6_20180420.csv')

    """ Get forward rates (theta) from Euribor """
    def get_interpolated_interest_rates(self, df_in : pd.DataFrame , frequency : float = 1/252, _print=False, method='from_derivatives'):
        df = df_in.copy(deep=True)
        _ending_tenor = df.index.get_level_values(1).sort_values()[-1]
        temp_df = df.loc[df.index.get_level_values(0) ==  df.index.get_level_values(0)[-1]].reset_index(level=0)[['rate']]
        old_index   = list(temp_df.index)
        new_index   = list(np.arange(0, _ending_tenor+frequency, frequency))
        temp_index  = pd.Index(new_index + old_index).sort_values().drop_duplicates()

        temp_df = temp_df.reindex(temp_index)
        temp_df = temp_df.astype(float)
        temp_df.sort_index(inplace=True)

        
        if _print:
            fig, ax = plt.subplots(4, figsize=(self.CHART_DIMENTIONS_DOUBLE_CHART[0], 18), sharex=True)
            ax[0].scatter(x=temp_df.index, y=temp_df.iloc[:,0].values, c='r')
            ax[1].scatter(x=temp_df.index, y=np.exp(-temp_df.iloc[:,0].values*temp_df.index.values), c='r')
            _temp_df = temp_df.interpolate(method='ffill')
            _temp_df = _temp_df.loc[new_index]
            _temp_df.plot(ax=ax[0], grid=True, title=f'Forward Filled Yield Curve - {_print}', \
                xlabel='Years - Until Maturity', ylabel='Interest Rate')
            ax[0].legend(['Given data points', 'Fitted Yield Curve'])
            np.exp(-_temp_df.iloc[:,0]*_temp_df.index.values).plot(ax=ax[1], grid=True, title=f'Forward Filled Discount Factors e^(-rt) - {_print}', \
                xlabel='Years - Until Maturity', ylabel='Discount Factors')
            ax[1].legend(['Given data points', 'Fitted Yield Curve'])
            
            ax[2].scatter(x=temp_df.index, y=temp_df.iloc[:,0].values, c='r')
            ax[3].scatter(x=temp_df.index, y=np.exp(-temp_df.iloc[:,0].values*temp_df.index.values), c='r')
            _temp_df = temp_df.interpolate(method=method)
            _temp_df = _temp_df.loc[new_index]
            _temp_df.plot(ax=ax[2], grid=True, title=f'Interpolated Yield Curve using Piecewise Polynomial - {_print}', \
                xlabel='Years - Until Maturity', ylabel='Interest Rate')
            ax[2].legend(['Given data points', 'Fitted Yield Curve'])
            np.exp(-_temp_df.iloc[:,0]*_temp_df.index.values).plot(ax=ax[3], grid=True, title=f'Interpolated Discount Factors e^(-rt) using Piecewise Polynomial - {_print}', \
                xlabel='Years - Until Maturity', ylabel='Discount Factors')
            ax[3].legend(['Given data points', 'Fitted Yield Curve'])
        
        temp_df.interpolate(method=method, inplace=True)
        temp_df = temp_df.loc[new_index]

        return temp_df.loc[new_index]

    def get_short_rate_HW(self, T: float, a: float, sigma: float,
                          rates_df: pd.DataFrame, paths: int = 1_000,
                          dt: float = 1/12, _print=False) -> np.ndarray:
        r_0       = rates_df.values[0, 0]      # spot short rate: f(0,0)
        intervals = floor(T / dt)
        f_0_t     = rates_df
    
        if _print:
            print(f'a = {a}, sigma = {sigma}, r_0 = {r_0:.6f}, T = {T}')
            print(f'intervals: {intervals} \nTotal Points: {paths*intervals}')
            print(f'paths: {paths} \ndt: {dt}')
    
        if _print: _t = perf_counter()
    
        # --- Sobol QMC normals (size = paths x intervals) ---
        sampler = qmc.Sobol(d=intervals, scramble=True)
        # using random() keeps the signature simple; clip avoids infs in ppf
        U = sampler.random(n=paths).clip(1e-12, 1-1e-12)
        W = norm.ppf(U)  # ~ N(0,1)
    
        # --- grid + g(t) ---
        t   = f_0_t.index.values[:intervals+1]
        g_t = f_0_t.values[:intervals+1, 0] + (sigma**2 / (2*a*a)) * (1 - np.exp(-a*t))**2
    
        hw_r_t      = np.zeros((paths, intervals+1))
        hw_r_t[:,0] = r_0
    
        if _print:
            print(f'hw.shape={hw_r_t.shape}, g_t.shape={g_t.shape}, t.shape={t.shape}')
    
        # exact discretization step
        exp_adt = np.exp(-a*dt)
        noise_scale = sigma * np.sqrt((1 - np.exp(-2*a*dt)) / (2*a))  # std of ∫ e^{-a(Δ-τ)} dW_τ
    
        for i in range(intervals):
            hw_r_t[:, i+1] = (
                hw_r_t[:, i] * exp_adt
                + g_t[i+1] - g_t[i] * exp_adt
                + noise_scale * W[:, i]
            )
    
        if _print:
            print(f'\n>>Time to simulate: {perf_counter()-_t:.3f} sec')
            return hw_r_t, g_t
    
        return hw_r_t

    def discount_factor_for_future_payments(self, forward_df: pd.DataFrame, short_rate_array: np.array, t:float, T :float, payment_frequency:float, 
                                                dt:float, alpha:float, sigma:float, return_t:bool, _print=False):
        discount_factor_array = np.exp(-forward_df.values[:,0] * forward_df.index.values)
        t =t + 0.000000001
        t_of_first_payment = ceil((t) / payment_frequency)*payment_frequency ; 
        number_of_payments_remaing = ceil((T-t)/payment_frequency) 
        starting_i  = floor(t / dt) #+ 1  # 2.1 / 0.5 = 4.2
        P_0_t       = discount_factor_array[starting_i]   # Zero-coupon bond starting at t=0, with maturity T
        f_0_t       = forward_df.values[starting_i, 0]
        r_t         = short_rate_array[:, starting_i]

        P_t_T_array = -np.ones((short_rate_array.shape[0], number_of_payments_remaing))
        if return_t: tenors = []
        if _print : 
            print(f't = {t}')
            print(f't_of_first_payment = {t_of_first_payment}')
            print(f'starting_i = {starting_i}')
            print(f'P_0_t = {P_0_t}')
            print(f'r_t.shape = {r_t.shape} ')   ; print(f'f_0_t shape = {f_0_t.shape}')
            print(f'number_of_payments_remaing = {number_of_payments_remaing}')
            print(f'discounted_rate_array.shape = {discount_factor_array.shape}')
            print(f'P_t_T_array.shape = {P_t_T_array.shape}')

        for payment_num in range(number_of_payments_remaing):
            if _print: print('------------ start evaluation -------------')
            
            ending_i    = floor((t_of_first_payment + payment_num*payment_frequency)/ dt) #+ 1
            _T          = t_of_first_payment + payment_num*payment_frequency
            P_0_T       = discount_factor_array[ending_i]     # Zero-coupon bond starting at t=0, with maturity t
            B_t_T       = (1/alpha) * (1 - np.exp(-alpha*(_T-t))) 

            if _print: 
                print(f'_T = {_T}')
                print(f'ending_i = {ending_i}')
                print(f'P_0_T = {P_0_T}')
                print(f'B_t_T = {B_t_T} , f_0_t = {f_0_t}')
                
            _a =   B_t_T*f_0_t                                          
            _b = (- sigma**2/(4*alpha)) * B_t_T**2 * (1-np.exp(-2*alpha*t))    
            _c = - B_t_T * r_t                                         
            if _print: print(f'_a = {_a} _a.shape = {_a.shape} \n_b = {_b} _b.shape = {_b.shape} \n_c = {_c} _c.shape = {_c.shape}')
            
            P_t_T = (P_0_T/P_0_t) * np.exp( _a + _b + _c)
            if _print: print(f'P_t_T.mean() = {P_t_T.mean()}')
            
            P_t_T_array[:,payment_num] = P_t_T
            if return_t: tenors.append(_T)
        
        if return_t:    return P_t_T_array, np.array(tenors)
        else:           return P_t_T_array


    """ IRS Valuation Simple """
    """ How much the receiver should pay """
    def interest_rate_swap_valuation(self, Notional:int, Floating_rate_df:pd.DataFrame, Floating_rate_cashflows_frequency:float, RFR_discounted_for_float_df:pd.DataFrame,\
        RFR_discounted_for_fixed_df:pd.DataFrame, Fixed_rate:float, Fixed_rate_cashflows_frequency:float):
        
        npv_floating = (Notional * Floating_rate_df * Floating_rate_cashflows_frequency * RFR_discounted_for_float_df)
        npv_fixed    = (Notional * Fixed_rate    * Fixed_rate_cashflows_frequency    * RFR_discounted_for_fixed_df)
        
        sum_npv_floating = np.sum(npv_floating)
        sum_npv_fix      = np.sum(npv_fixed)
        IRS_value        = sum_npv_floating - sum_npv_fix

        return npv_floating, npv_fixed, sum_npv_floating, sum_npv_fix, IRS_value


    """ IRS Valuation Vectorized """
    def interest_rate_swap_valuation_vectorised(self, Notional:int, Floating_rate:np.array, \
        Floating_rate_cashflows_frequency:float, RFR_discounted_for_float_list:list,\
        RFR_discounted_for_fixed_list:list, Fixed_rate:float, Fixed_rate_cashflows_frequency:float):

        # Vectorized version - for accepting HW paths
        npv_floating = (Notional * Floating_rate * Floating_rate_cashflows_frequency * RFR_discounted_for_float_list)
        npv_fixed    = (Notional * Fixed_rate    * Fixed_rate_cashflows_frequency    * RFR_discounted_for_fixed_list)
        
        sum_npv_floating = np.sum(npv_floating, axis=1)
        sum_npv_fix      = np.sum(npv_fixed, axis=1)
        IRS_value        = sum_npv_floating - sum_npv_fix

        return npv_floating, npv_fixed, sum_npv_floating, sum_npv_fix, IRS_value


    """ Get forward rates (theta) from Euribor """
    def __get_interpolated_cds_values(self, df_in : pd.DataFrame , frequency : float = 1/252, recovery:float=0.4, _print=False, method='from_derivatives'):
        _ending_tenor = df_in.index.sort_values()[-1]
        temp_df = df_in.copy(deep=True)
        old_index   = list(temp_df.index)
        new_index   = list(np.arange(0, _ending_tenor+frequency, frequency))
        temp_index  = pd.Index(new_index + old_index).sort_values().drop_duplicates()
        
        # if _print:
            # fig, ax = plt.subplots()
            # plt.scatter(x=temp_df.index, y=temp_df.iloc[:,0].values, c='r')

        temp_df = temp_df.reindex(temp_index)
        temp_df = temp_df.astype(float)
        temp_df.sort_index(inplace=True)
        
        temp_df.interpolate(method=method, inplace=True)
        temp_df = temp_df.loc[new_index]
        
        if _print:
            _PD_df = (temp_df / (1-recovery))
            _PD_df = (1 - _PD_df.shift(-1, fill_value=0)) * _PD_df * 100

            fig, ax = plt.subplots(figsize=self.CHART_DIMENTIONS_SINGLE_CHART)
            _PD_df.plot(ax=ax, figsize=self.CHART_DIMENTIONS_SINGLE_CHART, grid=True, title=_print, xlabel='Years - Maturity', ylabel='% Probability of Default')
            ax.scatter(x=old_index, y=_PD_df.loc[old_index].iloc[:,0].values, c='r')
            ax.legend(['Interpolated PD', 'Given data points'])
        return temp_df.loc[new_index]


    def get_rating_cds_interpolated_and_recovery(self, cds_ticker:str, dt:float, T:float, _print=False):
        if not (self.df_cds.Ticker == cds_ticker).sum() > 0:
            raise Exception('ERROR:Ticker Not found in given CDS dataset')

        _RECOVERY = self.df_cds.loc[self.df_cds.Ticker == cds_ticker, self.df_cds.columns[19]].values[0]
        subdf = self.df_cds.loc[self.df_cds.Ticker == cds_ticker, self.df_cds.columns[8:19]].T
        subdf.columns = ['CDS']
        subdf.index = [.5,1,2,3,4,5,7,10,15,20,30]
        subdf = pd.concat([subdf, pd.DataFrame([[0]], index=[0], columns=['CDS'])]).sort_index()
        cds_interpolated = self.__get_interpolated_cds_values(subdf, frequency=dt, recovery=_RECOVERY, _print=_print)
        if _print: 
            print(f'>>Recovery = {_RECOVERY}')
            return cds_interpolated.loc[cds_interpolated.index <= T].iloc[:,0], _RECOVERY
        else: return cds_interpolated.loc[cds_interpolated.index <= T].iloc[:,0], _RECOVERY


    def get_pd_estimate(self, Rating:str, Sector:str = None, Region: str = None, Country: str = None, Sector3x: str = None, return_df:bool=False):
        df = pd.read_csv('data/final_dataset.csv')
        df = df.loc[df.date == df.date.max()]

        df = df.loc[df.Rating == Rating] 
        if Sector : df = df.loc[df.Rating == Rating]
        if Region : df = df.loc[df.Region == Region]
        if Country : df = df.loc[df.Country == Country]
        if Sector3x : df = df.loc[df.Rating == Sector3x]

        if df.shape[0] == 0: raise Exception('ERROR:No records given the selected crietiria')
        mean_cds = df.cds_spread.mean()
        __5th_qt = df.cds_spread.quantile(0.05)
        _95th_qt = df.cds_spread.quantile(0.95)
        if return_df:
            return df[['cds_spread']]
        return mean_cds, __5th_qt, _95th_qt, df.cds_spread.std(), df.shape[0]


# get_rating_cds_interpolated_and_recovery(cds_ticker='BELG', dt=1/52, T=30, _print=False)
