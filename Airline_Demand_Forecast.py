#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import os
os.chdir("C:/Users/nisha/OneDrive/Desktop/Prgramming for BA 1")
 
def processing_data(data):
    data['days_prior']= pd.to_datetime(data.departure_date) - pd.to_datetime(data.booking_date)
    data['date_column'] = pd.to_datetime(data['booking_date'], format='%m/%d/%Y')
    data['day_of_week']=data.date_column.dt.day_name()
    final_demand= data.loc[data.departure_date == data.booking_date]
    final_demand['final_demand']=final_demand.cum_bookings
    data=data.merge(final_demand[['final_demand','departure_date']], left_on='departure_date',right_on='departure_date')
    return data
 
def calc_mase_number(data,forecast):
    data['abs_error']= abs(data.final_demand - forecast)
    data['abs_error_naive']= abs(data.final_demand - data.naive_fcst)
    sum_abs_error=data.loc[data['days_prior'] != '0 days', 'abs_error'].mean()
    sum_abs_error_naive=data.loc[data['days_prior'] != '0 days', 'abs_error_naive'].mean()
    mase=sum_abs_error / sum_abs_error_naive * 100
    return mase
 
def create_forecast_df(data,result):
    df=pd.DataFrame({'Departure Date':data.departure_date, 'Booking Date':data.booking_date,'Forecast':result})
    return df
def airlineForecast(training_data, validation_data):
    training = pd.read_csv(training_data, sep=',', header=0)
    validation = pd.read_csv(validation_data, sep=',', header=0)
    
    training=processing_data(training)
    training['remaining_demand']=training.final_demand - training.cum_bookings
    training['booking_rate']= training.cum_bookings / training.final_demand
    average_rem=training['remaining_demand'].groupby(training['days_prior']).mean()
    average_rate=training['booking_rate'].groupby(training['days_prior']).mean()
    result=pd.DataFrame({'days':average_rem.index,'average of remaining_demand':average_rem,'average of booking_rate':average_rate})
    avg_remm = training['remaining_demand'].groupby([training['days_prior'], training['day_of_week']]).mean().reset_index()
    avg_rate = training['booking_rate'].groupby([training['days_prior'], training['day_of_week']]).mean().reset_index()
    result_days = pd.DataFrame({'days_priorr': avg_rate['days_prior'],'days_week': avg_rate['day_of_week'],'average of rem': avg_remm['remaining_demand'],'average of rate': avg_rate['booking_rate']})
    
    validation=processing_data(validation)
    validation=validation.merge(result[['days','average of remaining_demand','average of booking_rate']],left_on='days_prior',right_on='days')
    validation=validation.merge(result_days[['days_priorr','days_week','average of rem','average of rate']],left_on=['days_prior','day_of_week'],right_on=['days_priorr','days_week'])
    validation=validation.sort_values(by=['departure_date','booking_date','days_prior'],ascending=[True,False,True])
    validation['demand_forecast_additive']= np.where(validation['days_prior'] != '0 days', validation['cum_bookings'] + validation['average of remaining_demand'], np.nan)
    validation['demand_forecast_multiplicative']= np.where(validation['days_prior'] != '0 days',validation.cum_bookings / validation['average of booking_rate'],np.nan)
    validation['demand_forecast_additive_days']= np.where(validation['days_prior'] != '0 days', validation['cum_bookings'] + validation['average of rem'], np.nan)
    validation['demand_forecast_multiplicative_days']= np.where(validation['days_prior'] != '0 days', validation['cum_bookings'] / validation['average of rate'], np.nan)
    mase_additive=calc_mase_number(validation,validation.demand_forecast_additive)
    mase_multiplicative=calc_mase_number(validation,validation.demand_forecast_multiplicative)
    mase_additive_days=calc_mase_number(validation,validation.demand_forecast_additive_days)
    mase_multiplicative_days=calc_mase_number(validation,validation.demand_forecast_multiplicative_days)
    
    fore_add=create_forecast_df(validation,validation.demand_forecast_additive)
    fore_mul=create_forecast_df(validation,validation.demand_forecast_multiplicative)
    fore_add_days=create_forecast_df(validation,validation.demand_forecast_additive_days)
    fore_mul_days=create_forecast_df(validation,validation.demand_forecast_multiplicative_days)
    
    mase_list=[mase_additive,mase_multiplicative,mase_additive_days,mase_multiplicative_days]
    forecast_list=[[mase_additive,fore_add],[mase_multiplicative,fore_mul],[mase_additive_days,fore_add_days],[mase_multiplicative_days,fore_mul_days]]
    min_mase=np.argmin(mase_list)
    
    training.to_csv('training.csv', sep=',', header=True, index=False)
    validation.to_csv('validation.csv', sep=',', header=True, index=False)
    return forecast_list[min_mase]
    
def main():
    train=input("Enter the filename of the training data")
    valid=input("Enter the filename of the validation data")
 
    print("\nThe Forecast from the Model with the Least MASE is:\n\n"+ str(airlineForecast(train, valid)))
main()

