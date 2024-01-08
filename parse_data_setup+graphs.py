# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:09:55 2024

@author: Gebruiker
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from CoolProp.CoolProp import PropsSI

mpl.rcParams['figure.dpi'] = 300
fontsz = 22
savefigs = True

def H(P,T,FLUID):
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
FILENAME = 'selected_data.csv'  #same directory as this python file
df = pd.read_csv(FILENAME, names=HEADER_REPLACE, skiprows=1, dtype=np.float64)

#Set working fluid and reference backend
BACKEND = 'REFPROP' 
FLUID1 = 'R134a'     
FLUID=BACKEND+'::'+FLUID1  

#calculate flow from pump
d_piston = 40e-3 #[mm]
d_rod = 16e-3 #[mm]
Area_Rod = np.pi*( (d_piston/2)**2 - (d_rod/2)**2 ) # [m2]
Area_No_Rod = np.pi*(d_piston/2)**2 # [m2]
Area_pump = Area_Rod + Area_No_Rod #[m2]
s_pump = 40e-3 # [m]
cycle_time_pump = 4 # [s] //Time required for a full cycle
df['flow_WF_pump'] = Area_pump * s_pump / cycle_time_pump # [m3/s]
df['rho_WF_pump'] = PropsSI('D','P',np.array(df.P3*1e5),'T',np.array(df.T8+273.15),FLUID)

#calculate flow from driver
s_driver = 92e-3 #[m]
d_driver = 80e-3 #[m]
Area_driver = 2 * np.pi*(d_driver/2)**2 # [m2]
cycle_time_driver = 4 # [s] //Time required for a full cycle
df['flow_WF_driver'] = Area_driver * s_driver / cycle_time_driver #[m3/s]
df['rho_WF_driver'] = PropsSI('D','P',np.array(df.P1b*1e5),'T',np.array(df.T1+273.15),FLUID)

#calculate flow from sensor
df['flow_WF_sensor'] = df.flow2.mean() / 60 / 1000
df['rho_WF_sensor'] = PropsSI('D','P',np.array(df.P3*1e5),'T',np.array(df.T8+273.15),FLUID)

df['W'] = df.flow * df.P4 * 1000 / 600 #Power output W
df['RPM'] = 1000 * df.flow / 5.54 #RPM from volume flow

#Mass flow working fluid
df['m_dotWF_pump'] = df.rho_WF_pump * df.flow_WF_pump #[kg/s]
df['m_dotWF_driver'] = df.rho_WF_driver * df.flow_WF_driver #[kg/s]
df['m_dotWF_sensor'] = df.rho_WF_sensor * df.flow_WF_sensor #[kg/s]

#Heat input
H_heaterIN = H(df.P1b,df.T2,FLUID)
H_heaterOUT = H(df.P1b,df.T1,FLUID)
df['Qin_pump'] = df.m_dotWF_pump * (H_heaterOUT - H_heaterIN) # [W]
df['Qin_driver'] = df.m_dotWF_driver * (H_heaterOUT - H_heaterIN) # [W]
df['Qin_sensor'] = df.m_dotWF_sensor * (H_heaterOUT - H_heaterIN) # [W]

#Pump work 
H_pumpIN = H(df.P3,df.T8,FLUID)
H_pumpOUT = H(df.P8,df.T4,FLUID)
df['W_p_pump'] =  df.m_dotWF_pump * (H_pumpOUT - H_pumpIN) #[W]
df['W_p_driver'] =  df.m_dotWF_driver * (H_pumpOUT - H_pumpIN) #[W]
df['W_p_sensor'] =  df.m_dotWF_sensor * (H_pumpOUT - H_pumpIN) #[W]

#Delta T over pump (should be very small)
df['dT_p'] = df.T4-df.T8 #[°C]

#Energy change RECEIVER flow regeneration
# df['dHr'] = m_dotWF * (H(df.P1b,df.T2,FLUID) - H(df.P8,df.T4,FLUID)) #[W]
# #Energy change DONOR flow regeneration
# df['dHd'] = m_dotWF * (H(df.P2,df.T3,FLUID) - H(df.P3,df.T5,FLUID)) #[W]

#Efficiencies
df['eta_pump'] = (df.W - df.W_p_pump) / df.Qin_pump
df['eta_driver'] = (df.W - df.W_p_driver) / df.Qin_driver
df['eta_sensor'] = (df.W - df.W_p_sensor) / df.Qin_sensor
df['eta_c'] = 1 - (df.T8+273.15)/(df.T1+273.15)  
df['eta_sp'] = df['eta_sensor']/df['eta_c']


window_size=10
periods=1
#smoothed_df = df.copy()
for column in df.columns:
    df[column] = df[column].rolling(window=window_size, min_periods=periods).mean()
    
#plotting
L = len(df) 

fig1, ax1 = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax1.plot(df['eta_pump'], label='Efficiency Pump')
ax1.plot(df['eta_driver'], label='Efficiency Driver')
ax1.plot(df['eta_sensor'], label='Efficiency Sensor')
ax1.plot(df['eta_c'],'k--',label='Carnot')
ax1.set_xlabel(r'$index$',fontsize=fontsz)
ax1.set_ylabel(r"$\eta$",fontsize=fontsz)
#ax1.set_xlim(start_index, end_index)
ax1.set_ylim(0.0,0.2)
ax1.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax1.legend(fontsize=fontsz-5,loc='best')
ax1.grid()
if savefigs: plt.savefig('GRAPHS\eta_60.png')
#st.pyplot(fig1)

fig2a, ax2a = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax2a.plot(df['flow_WF_pump']*60*1000, label='Volume Flow WF PUMP')
ax2a.plot(df['flow_WF_driver']*60*1000, label='Volume Flow WF DRIVER')
ax2a.plot(df['flow_WF_sensor']*60*1000, label='Volume Flow WF SENSOR')
ax2a.set_xlabel(r'$index$',fontsize=fontsz)
ax2a.set_ylabel(r"Volume Flow WF [L/min]",fontsize=fontsz)
#ax2a.set_xlim(start_index, end_index)
#ax2.set_ylim(0.0,0.2)
ax2a.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax2a.legend(fontsize=fontsz-5,loc='best')
ax2a.grid()
if savefigs: plt.savefig('GRAPHS\V_flow_60.png')
#st.pyplot(fig2a)

fig2b, ax2b = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax2b.plot(df['rho_WF_pump'], label='Density WF PUMP')
ax2b.plot(df['rho_WF_driver'], label='Density WF DRIVER')
ax2b.plot(df['rho_WF_sensor'], label='Density WF SENSOR')
ax2b.set_xlabel(r'$index$',fontsize=fontsz)
ax2b.set_ylabel(r"Density WF [kg/m3]",fontsize=fontsz)
#ax2b.set_xlim(start_index, end_index)
#ax2.set_ylim(0.0,0.2)
ax2b.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax2b.legend(fontsize=fontsz-5,loc='best')
ax2b.grid()
if savefigs: plt.savefig('GRAPHS\\rho_60.png')
#st.pyplot(fig2b)

fig2c, ax2c = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax2c.plot(df['m_dotWF_pump']*60, label='Mass Flow WF PUMP')
ax2c.plot(df['m_dotWF_driver']*60, label='Mass Flow WF DRIVER')
ax2c.plot(df['m_dotWF_sensor']*60, label='Mass Flow WF SENSOR')
ax2c.set_xlabel(r'$index$',fontsize=fontsz)
ax2c.set_ylabel(r"Mass Flow WF [kg/min]",fontsize=fontsz)
#ax2c.set_xlim(start_index, end_index)
#ax2.set_ylim(0.0,0.2)
ax2c.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax2c.legend(fontsize=fontsz-5,loc='best')
ax2c.grid()
if savefigs: plt.savefig('GRAPHS\massflow_60.png')
#st.pyplot(fig2c)

fig3, ax3 = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax3.plot(df['W'], label='Work Output')
ax3.plot(df['W_p_pump'], label='Pump Work Pump')
ax3.plot(df['W_p_driver'], label='Pump Work Driver')
ax3.plot(df['W_p_sensor'], label='Pump Work Sensor')
ax3.set_xlabel(r'$index$',fontsize=fontsz)
ax3.set_ylabel(r"Work [W]",fontsize=fontsz)
#ax3.set_xlim(start_index, end_index)
#ax3.set_ylim(0.0,0.2)
ax3.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax3.legend(fontsize=fontsz-5,loc='best')
ax3.grid()
if savefigs: plt.savefig('GRAPHS\work_60.png')
#st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax4.plot(df['Qin_pump'], label='Heat Input Pump')
ax4.plot(df['Qin_driver'], label='Heat Input Driver')
ax4.plot(df['Qin_sensor'], label='Heat Input Sensor')
ax4.set_xlabel(r'$index$',fontsize=fontsz)
ax4.set_ylabel(r"Heat Input [W]",fontsize=fontsz)
#ax4.set_xlim(start_index, end_index)
#ax4.set_ylim(0.0,0.2)
ax4.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax4.legend(fontsize=fontsz-5,loc='best')
ax4.grid()
if savefigs: plt.savefig('GRAPHS\Qin_60.png')
#st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax5.plot(df['T1'], label='T1 Heater OUT')
ax5.plot(df['T2'], label='T2 Heater IN')
ax5.plot(df['T3'], label='T3 Engine OUT')
ax5.plot(df['T4'], label='T4 Pump OUT')
ax5.plot(df['T5'], label='T5 Cooler IN')
ax5.plot(df['T6'], label='T6 Heating Water OUT')
ax5.plot(df['T7'], label='T7 Cooling Water OUT')
ax5.plot(df['T8'], label='T8 Cooler OUT')
ax5.set_xlabel(r'$index$',fontsize=fontsz)
ax5.set_ylabel(r"Temperatures [°C]",fontsize=fontsz)
#ax5.set_xlim(start_index, end_index)
#ax1.set_ylim(0.0,0.2)
ax5.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax5.legend(fontsize=fontsz-10,loc='best')
ax5.grid()
if savefigs: plt.savefig('GRAPHS\T_60.png')
#st.pyplot(fig5)

fig6, ax6 = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax6.plot(df['P1b'], 'b--', label='P1 Engine IN')
ax6.plot(df['P2'], label='P2 Engine OUT')
ax6.plot(df['P3'], label='P3 Pump IN')
ax6.plot(df['P4'], label='P4 Oil')
ax6.plot(df['P5'], label='P5 R Chamber')
ax6.plot(df['P6'], label='P6 L Chamber')
ax6.plot(df['P8'], label='P8 Pump OUT')
ax6.set_xlabel(r'$index$',fontsize=fontsz)
ax6.set_ylabel(r"Pressures [barg]",fontsize=fontsz)
#ax6.set_xlim(start_index, end_index)
#ax1.set_ylim(0.0,0.2)
ax6.tick_params(axis='both', which='major', labelsize=fontsz-5) 
ax6.legend(fontsize=fontsz-10,loc='best')
ax6.grid()
if savefigs: plt.savefig('GRAPHS\P_60.png')
#st.pyplot(fig6)











