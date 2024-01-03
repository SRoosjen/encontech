import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from CoolProp.CoolProp import PropsSI
import math

print(os.getcwd())



df = pd.read_excel(r'C:\Users\Lenovo\.spyder-py3\EXP111.xlsx')

#t = df['t'].tolist()

P1  = df['P1b=engine in Ave. (bar)'].tolist() #P_High
P2  = df['P2=engine out Ave. (bar)'].tolist()
P3  = df['P3=f. pump in Ave. (bar)'].tolist() #P_Low
P4  = df['P4=oil Ave. (bar)'].tolist()
P5  = df['P5=right chamber Ave. (bar)'].tolist()
P6  = df['P6=left chamber Ave. (bar)'].tolist()
P8  = df['P8=f. pump out Ave. (bar)'].tolist()  
RPM  = df['RPM Ave. (V)'].tolist()
flow  = df['flowmeter average Ave. (Lmin)'].tolist() #Oil flow
T1  = df['T1=heater out Ave. (C)'].tolist() #T_High
T2  = df['T2=heater in Ave. (C)'].tolist()
T3  = df['T3=engine out Ave. (C)'].tolist()
T4   = df['T4=pump out Ave. (C)'].tolist()
T5  = df['T5=cooler in Ave. (C)'].tolist()
T6  = df['T6=Heating water out Ave. (C)'].tolist()
T7  = df['T7=cooling water out Ave. (C)'].tolist()
T8  = df['T8=cooler out Ave. (C)'].tolist() #T_Low
flow2 = df['flowmeter (Honsberg) Ave. (Lmin)'].tolist() #Refflow


t = np.linspace(0,len(P1),len(P1))

P1=np.array(P1)
P3=np.array(P3)
DP = P1 - P3
P4=np.array(P4)
P5=np.array(P5)
P6=np.array(P6)
P8=np.array(P8)
T1=np.array(T1)
T2=np.array(T2)
T3=np.array(T3)
T4=np.array(T4)
T5=np.array(T5)
T6=np.array(T6)
T7=np.array(T7)
T8=np.array(T8)
F_oil=np.array(flow)
P_Oil = np.array(P4)
F_ref = np.array(flow2)


#Power=F_oil*100*P_Oil/60 #Power output

#General data plots
plt.figure()
plt.plot(t,T1, label='Heater out')
plt.plot(t,T2, label='Heater in')
plt.plot(t,T3, label='Engine out') 
plt.plot(t,T4, label='Pump out')   
plt.plot(t,T5, label='Cooler in') 
plt.plot(t,T6, label='Heating water out ') 
plt.plot(t,T7, label='Cooling water out') 
plt.plot(t,T8, label='Cooler out') 
plt.ylabel('Data')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('Data logged')
plt.show()


plt.figure()
plt.plot(t,P1, label='Engine in') 
plt.plot(t,P2, label='Engine out')
plt.plot(t,P3, label='Pump in')
plt.plot(t,P8, label='Pump out')
plt.plot(t,P4, label='Oil')
#plt.plot(t,P5, label='P5-right chamber')
#plt.plot(t,P6, label='P6-left chamber')
plt.ylabel('Pressure (bar)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
#plt.title('FI-01')
plt.show()


plt.figure()
plt.plot(t,T1, label='T_High') 
plt.plot(t,P2, label='T_Low')
plt.ylabel('Tempeature ($\degree C$)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
#plt.title('FI-01')
plt.show()

plt.figure()
plt.plot(t,F_oil, label='Oil Flow') 
plt.plot(np.linspace(0,len(F_ref),len(F_ref)),F_ref, label='R134A')
plt.ylabel('Flow (l/min')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
#plt.title('FI-01')
plt.show()




# Define the time range
t_min = 8900#6150#3000
t_max = 9700#7000#3700

# Find indices for temperatures within the specified range
indices_range = np.where((t >= t_min) & (t <= t_max))[0]

# Separate data based on time range
range_data = {
    'T_H': T1[indices_range],
    'T_L': T8[indices_range],
    'P_H': P1[indices_range],
    'P_L': P3[indices_range],
    'F_oil': F_oil[indices_range],
    'P_oil': P4[indices_range],
    'P_po': P8[indices_range],
    'T_po': T4[indices_range],
    'F_ref':F_ref[indices_range],
    'T_hin':T2[indices_range],
    'T_cwater':T7[indices_range]
    
    
}

# Save each dataset in new NumPy arrays
T_H = range_data['T_H']
T_L = range_data['T_L']
P_H = range_data['P_H']
P_L = range_data['P_L']
F_oil = range_data['F_oil']
F_ref = range_data['F_ref']
P_oil = range_data['P_oil']
P_po = range_data['P_po']
T_po = range_data['T_po']
T_hin = range_data['T_hin']
T_cwater = range_data['T_cwater']

#Set rolling average window
w = 15
#Calculating rolling average for the dataset
T_HRA = np.convolve(T_H, np.ones(w)/w, mode='valid')
T_LRA = np.convolve(T_L, np.ones(w)/w, mode='valid')
P_HRA = np.convolve(P_H, np.ones(w)/w, mode='valid')
P_LRA = np.convolve(P_L, np.ones(w)/w, mode='valid')
P_oilRA = np.convolve(P_oil, np.ones(w)/w, mode='valid')
F_oilRA = np.convolve(F_oil, np.ones(w)/w, mode='valid')
F_refRA = np.convolve(F_ref, np.ones(w)/w, mode='valid')
P_poRA = np.convolve(P_po, np.ones(w)/w, mode='valid')
T_poRA = np.convolve(T_po, np.ones(w)/w, mode='valid')
T_hinRA = np.convolve(T_hin, np.ones(w)/w, mode='valid')
T_cwaterRA = np.convolve(T_cwater, np.ones(w)/w, mode='valid')


# Create a time array corresponding to the rolling average data
tt = t[:len(T_HRA)]

#Create time array corresponding to original data
t = np.linspace(0,len(P_H),len(P_H))

Power=F_oilRA*100*P_oilRA/60 #Power output W
RPM = 1000*F_oilRA/5.54 


#R134a mass flow
m_ref = np.zeros(len(tt))
m_ref22 = np.zeros(len(tt))
A = np.average(P_poRA) #Pressure pump out
B = np.average(T_poRA) #temperature pump out
rho = PropsSI('Dmass','T',B+273.15,'P',A*pow(10,5),'R134a')

#Calculate mass flow from flowmeter
m_ref22 = (F_refRA*rho/1000)/60 #R134a kg/s

#calculate flow from pump
Area = math.pi*((40-16)/2)**2 #mm2
s = 50 #mm
time = 4 #s 
f = 10**(-6)*Area*s/time #(l/s)
flowpump = np.full(len(tt),f)

m_ref = flowpump*rho/1000 #kg/s




#Q heater
Q_h = np.zeros(len(tt))
C = np.average(T_HRA) #temperature high or heater out
D = np.average(P_HRA) #pressure high or engine in
E = np.average(T_hinRA) #temperature heater in
H1 = PropsSI('H','T',C+273.15,'P',D*pow(10,5),'R134a')
H2 = PropsSI('H','T',E+273.15,'P',D*pow(10,5),'R134a')
#print(H1)
#print(H2)
Q_h = m_ref*(H1-H2)#/60 #W

y = 3.1558*(t_max-t_min)/3600 +0.3335 #kWh elec heater consumption
Q_h2 = 1000*y/((t_max-t_min)/3600)

Q_h22 = np.full(len(tt),Q_h2)

#W pump
W_p = np.zeros(len(tt))
H3 = PropsSI('H','T',B+273.15,'P',A*pow(10,5),'R134a')
F = np.average(P_LRA) #pressure low or pump in
G = np.average(T_LRA) #temperature low or cooler out
H4 = PropsSI('H','T',G+273.15,'P',F*pow(10,5),'R134a')
W_p = m_ref*(H3-H4)#/60 #W


eff = 100*(Power-W_p)/Q_h #efficiency based on theoretical heat input

eff2 = 100*(Power-W_p)/Q_h22 #efficiency based on electrical heater consumption

Carnoteff = (1-T_LRA/T_HRA)*100





#Plots of data used
plt.figure()
plt.plot(tt,T_LRA, label='cooler out or pump in') 
plt.plot(tt,T_poRA,label='pump out')
plt.ylabel('Tempeature ($\degree C$)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('Refrigerant heating in the pump')
plt.show()

plt.figure()
plt.plot(t,T_cwater) 
plt.ylabel('Tempeature ($\degree C$)')
plt.xlabel('Time (s)')
#plt.legend()
plt.grid ()
plt.title('Cooling water')
plt.show()


#plots before and after rolling average
plt.figure()
plt.plot(t,T_H, label='Original') 
plt.plot(tt,T_HRA,label='Rolling Average')
plt.ylabel('Tempeature ($\degree C$)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('T_High (heater out)')
plt.show()
 
plt.figure()
plt.plot(t,T_L, label='Original') 
plt.plot(tt,T_LRA,label='Rolling Average')
plt.ylabel('Tempeature ($\degree C$)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('T_Low (cooler out)')
plt.show()

plt.figure()
plt.plot(t,T_po, label='Original') 
plt.plot(tt,T_poRA,label='Rolling Average')
plt.ylabel('Tempeature ($\degree C$)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('T_pump,out')
plt.show()

plt.figure()
plt.plot(t,T_hin, label='Original') 
plt.plot(tt,T_hinRA,label='Rolling Average')
plt.ylabel('Tempeature ($\degree C$)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('T_heater,in')
plt.show()

plt.figure()
plt.plot(t,P_H, label='Original') 
plt.plot(tt,P_HRA,label='Rolling Average')
plt.ylabel('Pressure (bar)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('P_High (engine in)')
plt.show()
 
plt.figure()
plt.plot(t,P_L, label='Original') 
plt.plot(tt,P_LRA,label='Rolling Average')
plt.ylabel('Pressure (bar)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('P_Low (pump in)')
plt.show()

plt.figure()
plt.plot(t,P_oil, label='Original') 
plt.plot(tt,P_oilRA,label='Rolling Average')
plt.ylabel('Pressure (bar)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('P_oil')
plt.show()


plt.figure()
plt.plot(tt,P_HRA-P_LRA, label='DP_R134a') 
plt.plot(tt,P_oilRA,label='P_Oil')
plt.ylabel('Pressure (bar)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('DP_R134a vs P_oil relation')
plt.show()


plt.figure()
plt.plot(t,P_po, label='Original') 
plt.plot(tt,P_poRA,label='Rolling Average')
plt.ylabel('Pressure (bar)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('P_pump,out')
plt.show()

plt.figure()
plt.plot(t,F_oil, label='Original') 
plt.plot(tt,F_oilRA,label='Rolling Average')
plt.ylabel('Flow (l/min)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('Oil flowrate')
plt.show()

plt.figure()
plt.plot(t,F_ref, label='Original') 
plt.plot(tt,F_refRA,label='Rolling Average')
plt.ylabel('Flow (l/min)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid ()
plt.title('R134a flowrate')
plt.show()


#Calculation plots    

plt.figure()
plt.plot(tt,flowpump*60,label='pump')
plt.plot(tt,F_refRA,label='flowmeter')
plt.plot() 
plt.ylabel('R134a (l/min))')
plt.xlabel('Time (s)')
plt.grid ()
plt.title('R134a  flowrate')
plt.legend()
plt.show()




plt.figure()
plt.plot(tt,m_ref,label='pump')
plt.plot(tt,m_ref22,label='flowmeter')
plt.plot() 
plt.ylabel('m_R134a (kg/s))')
plt.xlabel('Time (s)')
plt.grid ()
plt.title('R134a mass flowrate')
plt.legend()
plt.show()



#Calculation plots    
plt.figure()
plt.plot(tt,Power) 
plt.ylabel('Power (W)')
plt.xlabel('Time (s)')
plt.grid ()
plt.title('Power output')
plt.show()        
 
plt.figure()
plt.plot(tt,RPM) 
plt.ylabel('RPM')
plt.xlabel('Time (s)')
plt.grid ()
plt.title('Hydraulic motor RPM')
plt.show()     
 
plt.figure()
plt.plot(tt,Q_h) 
plt.ylabel('Power (W)')
plt.xlabel('Time (s)')
plt.grid ()
plt.title('Theoretical heat input')
plt.show()

plt.figure()
plt.plot(tt,W_p) 
plt.ylabel('Power (W)')
plt.xlabel('Time (s)')
plt.grid ()
plt.title('Pump work')
plt.show()


data = {'Theoretical heat input calculation (1)':np.average(Q_h), 'Electrical heater consumption (2)':Q_h2}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon', 
        width = 0.4)
 
#plt.xlabel("Courses offered")
plt.ylabel("Power (W)")
plt.title("Q_heater")
plt.show()



plt.figure()
plt.plot(tt,eff,label='1') 
plt.plot(tt,eff2,label='2') 
plt.ylabel('n (%)')
plt.xlabel('Time (s)')
plt.grid ()
plt.legend()
plt.title('Efficiency')
plt.show()

plt.figure()
plt.plot(tt,Carnoteff) 

plt.ylabel('n (%)')
plt.xlabel('Time (s)')
plt.grid ()

plt.title('Carnot Efficiency')
plt.show()

data = {'Eff/Carnot eff (1)':100*np.average(eff)/np.average(Carnoteff), 'Eff/Carnot eff (2)':100*np.average(eff2)/np.average(Carnoteff)}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon', 
        width = 0.4)
 
#plt.xlabel("Courses offered")
plt.ylabel("(%)")
plt.title("Efficiency as a percentage of Carnot efficiency")
plt.show()


