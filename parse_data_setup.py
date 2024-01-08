# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:09:55 2024

@author: Gebruiker
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

def H(P,T):
    return PropsSI('H','P',np.array(P*1e5),'T',np.array(T+273.15),FLUID)
    
HEADER_REPLACE = \
        ['time', 'P2', 'P1b',
       'P3', 'P4',
       'P5', 'P6',
       'RPM', 'P8',
       'flow2', 'RPMavg',
       'flow', 'T1',
       'T2', 'T3',
       'T4', 'T5',
       'T6', 'T7',
       'T8', 'dP']

#Read csv file, if custom header skiprow=1
FILENAME = 'EXP111.csv'  #same directory as this python file
df = pd.read_csv(FILENAME, names=HEADER_REPLACE, skiprows=1, dtype=np.float64)

#calculate flow from pump
d_piston = 40e-3 #[mm]
d_rod = 16e-3 #[mm]
Area_Rod = np.pi*( (d_piston/2)**2 - (d_rod/2)**2 ) # [m2]
Area_No_Rod = np.pi*(d_piston/2)**2 # [m2]
Area_pump = Area_Rod + Area_No_Rod #[m2]
s_pump = 40e-3 # [m]
cycle_time_pump = 4 # [s] //Time required for a full cycle
flow_WF_pump = Area_pump * s_pump / cycle_time_pump # [m3/s]

#calculate flow from driver
s_driver = 92e-3 #[m]
d_driver = 80e-3 #[m]
Area_driver = 2 * np.pi*(d_driver/2)**2 # [m2]
cycle_time_driver = 4 # [s] //Time required for a full cycle
flow_WF_driver = Area_driver * s_driver / cycle_time_driver #[m3/s]

#choose the volume flow
flow_WF = flow_WF_driver

#Set working fluid and reference backend
BACKEND = 'REFPROP' 
FLUID1 = 'R134a'     
FLUID=BACKEND+'::'+FLUID1  

df['W'] = df.flow * df.P4 * 1000 / 600 #Power output W
df['RPM'] = 1000 * df.flow / 5.54 #RPM from volume flow

#Mass flow working fluid
rho_WF = PropsSI('D','P',np.array(df.P1b*1e5),'T',np.array(df.T1+273.15),FLUID)    
df['m_dotWF'] = rho_WF * flow_WF #[kg/s]

#Heat input
H_heaterIN = H(df.P1b,df.T2)
H_heaterOUT = H(df.P1b,df.T1)
df['Qin'] = df.m_dotWF * (H_heaterOUT - H_heaterIN) # [W]

#Pump work 
H_pumpIN = H(df.P3,df.T8)
H_pumpOUT = H(df.P8,df.T4)
df['W_p'] =  df.m_dotWF * (H_pumpOUT - H_pumpIN) #[W]

#Delta T over pump (should be very small)
df['dT_p'] = df.T4-df.T8 #[°C]

#Energy change RECEIVER flow regeneration
df['dHr'] = flow_WF * (H(df.P1b,df.T2) - H(df.P8,df.T4)) #[W]
#Energy change DONOR flow regeneration
df['dHd'] = flow_WF * (H(df.P2,df.T3) - H(df.P3,df.T5)) #[W]

#Efficiencies
df['eta'] = (df.W - df.W_p) / df.Qin
df['eta_c'] = 1 - (df.T8+273.15)/(df.T1+273.15)  
df['eta_sp'] = df['eta']/df['eta_c']


window_size=500
periods=1
#smoothed_df = df.copy()
for column in df.columns:
    df[column] = df[column].rolling(window=window_size, min_periods=periods).mean()


