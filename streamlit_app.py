import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

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

def main():
    st.title("Data Visualization App")
    # Sidebar for adjustable variables
    st.sidebar.header("Adjustable Variables")

    # Create a file uploader widget
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    # Adjustable parameters
    window_size = st.sidebar.slider("Smoothing Window Size", 1, 2000, 5)

    # Check if a file is uploadedr'EXP111.csv'
    if uploaded_file is not None:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file, names=HEADER_REPLACE, skiprows=1, dtype=np.float64)
        #df = pd.read_csv(uploaded_file)

        # Smooth the data
        smoothed_df = smooth_data(df, window_size)
    
        # Choose a range of data to display
        start_index, end_index = get_data_range(smoothed_df)
    
        # Dropdown multiselect for selecting columns to be plotted
        selected_columns = st.sidebar.multiselect("Select Columns to Plot", smoothed_df.columns)
        if not selected_columns:
            st.warning("Please select at least one column.")
    
        # Plot the selected data for selected columns with bounds
        plot_selected_data_with_bounds(smoothed_df, start_index, end_index, selected_columns)
    
        # Plot the data between the selected vertical lines for selected columns
        plot_selected_data(smoothed_df, start_index, end_index, selected_columns)
        
        
        selected_data = smoothed_df.iloc[start_index:end_index + 1]
        parsed_data = parsecsv(selected_data,start_index, end_index)
        plot_parsed_data(parsed_data,start_index, end_index)

def plot_parsed_data(df,start_index, end_index): 
    st.subheader("VISU TEST")
    fontsz = 22
    L = len(df) 

    
    fig1, ax1 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax1.plot(df['eta'], label='Efficiency')
    ax1.plot(df['eta_c'],'k--',label='Carnot')
    ax1.set_xlabel(r'$index$',fontsize=fontsz)
    ax1.set_ylabel(r"$\eta$",fontsize=fontsz)
    ax1.set_xlim(start_index, end_index)
    ax1.set_ylim(0.0,0.2)
    ax1.tick_params(axis='both', which='major', labelsize=fontsz-5) 
    ax1.legend(fontsize=fontsz-5,loc='best')
    ax1.grid()
    st.pyplot(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax2.plot(df['m_dotWF_pump']*60, label='Mass Flow WF PUMP')
    ax2.plot(df['m_dotWF_driver']*60, label='Mass Flow WF DRIVER')
    ax2.plot(df['m_dotWF_sensor']*60, label='Mass Flow WF SENSOR')
    ax2.set_xlabel(r'$index$',fontsize=fontsz)
    ax2.set_ylabel(r"Mass Flow WF [kg/min]",fontsize=fontsz)
    ax2.set_xlim(start_index, end_index)
    #ax2.set_ylim(0.0,0.2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsz-5) 
    ax2.legend(fontsize=fontsz-5,loc='best')
    ax2.grid()
    st.pyplot(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax3.plot(df['W'], label='Work Output')
    ax3.plot(df['W_p'], label='Work Pump')
    ax3.set_xlabel(r'$index$',fontsize=fontsz)
    ax3.set_ylabel(r"Work [W]",fontsize=fontsz)
    ax3.set_xlim(start_index, end_index)
    #ax3.set_ylim(0.0,0.2)
    ax3.tick_params(axis='both', which='major', labelsize=fontsz-5) 
    ax3.legend(fontsize=fontsz-5,loc='best')
    ax3.grid()
    st.pyplot(fig3)
    
    fig4, ax4 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax4.plot(df['Qin'], label='Heat Input')
    ax4.set_xlabel(r'$index$',fontsize=fontsz)
    ax4.set_ylabel(r"Heat Input [W]",fontsize=fontsz)
    ax4.set_xlim(start_index, end_index)
    #ax4.set_ylim(0.0,0.2)
    ax4.tick_params(axis='both', which='major', labelsize=fontsz-5) 
    ax4.legend(fontsize=fontsz-5,loc='best')
    ax4.grid()
    st.pyplot(fig4)
    
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
    ax5.set_xlim(start_index, end_index)
    #ax1.set_ylim(0.0,0.2)
    ax5.tick_params(axis='both', which='major', labelsize=fontsz-5) 
    ax5.legend(fontsize=fontsz-10,loc='best')
    ax5.grid()
    st.pyplot(fig5)
    
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
    ax6.set_xlim(start_index, end_index)
    #ax1.set_ylim(0.0,0.2)
    ax6.tick_params(axis='both', which='major', labelsize=fontsz-5) 
    ax6.legend(fontsize=fontsz-10,loc='best')
    ax6.grid()
    st.pyplot(fig6)

    return

def parsecsv(df,start_index, end_index):
    # Input box to get user-specified filename
    #file_name = st.text_input("Enter file name (including extension):", "selected_data.csv")

    # csv_file = dataframe.to_csv(index=False)
    # b64 = base64.b64encode(csv_file.encode()).decode()
    # href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download Selected Data</a>'
    # st.markdown(href, unsafe_allow_html=True)
    #calculate flow from pump
    
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
    flow_WF_pump = Area_pump * s_pump / cycle_time_pump # [m3/s]
    rho_WF_pump = PropsSI('D','P',np.array(df.P3*1e5),'T',np.array(df.T8+273.15),FLUID)

    #calculate flow from driver
    s_driver = 92e-3 #[m]
    d_driver = 80e-3 #[m]
    Area_driver = 2 * np.pi*(d_driver/2)**2 # [m2]
    cycle_time_driver = 4 # [s] //Time required for a full cycle
    flow_WF_driver = Area_driver * s_driver / cycle_time_driver #[m3/s]
    rho_WF_driver = PropsSI('D','P',np.array(df.P1b*1e5),'T',np.array(df.T1+273.15),FLUID)

    #calculate flow from sensor
    flow_WF_sensor = df.flow2.mean() / 60 / 1000
    rho_WF_sensor = PropsSI('D','P',np.array(df.P3*1e5),'T',np.array(df.T8+273.15),FLUID)

    df['W'] = df.flow * df.P4 * 1000 / 600 #Power output W
    df['RPM'] = 1000 * df.flow / 5.54 #RPM from volume flow

    #Mass flow working fluid
    df['m_dotWF_pump'] = rho_WF_pump * flow_WF_pump #[kg/s]
    df['m_dotWF_driver'] = rho_WF_driver * flow_WF_driver #[kg/s]
    df['m_dotWF_sensor'] = rho_WF_sensor * flow_WF_sensor #[kg/s]

    #choose the mass/volume flow [make option in sidepanel]
    m_dotWF = np.array(df['m_dotWF_driver'])
    
    #Heat input
    H_heaterIN = H(df.P1b,df.T2,FLUID)
    H_heaterOUT = H(df.P1b,df.T1,FLUID)
    df['Qin'] = m_dotWF * (H_heaterOUT - H_heaterIN) # [W]

    #Pump work 
    H_pumpIN = H(df.P3,df.T8,FLUID)
    H_pumpOUT = H(df.P8,df.T4,FLUID)
    df['W_p'] =  m_dotWF * (H_pumpOUT - H_pumpIN) #[W]

    #Delta T over pump (should be very small)
    df['dT_p'] = df.T4-df.T8 #[°C]

    #Energy change RECEIVER flow regeneration
    df['dHr'] = m_dotWF * (H(df.P1b,df.T2,FLUID) - H(df.P8,df.T4,FLUID)) #[W]
    #Energy change DONOR flow regeneration
    df['dHd'] = m_dotWF * (H(df.P2,df.T3,FLUID) - H(df.P3,df.T5,FLUID)) #[W]

    #Efficiencies
    df['eta'] = (df.W - df.W_p) / df.Qin
    df['eta_c'] = 1 - (df.T8+273.15)/(df.T1+273.15)  
    df['eta_sp'] = df['eta']/df['eta_c']
    return df
        

def smooth_data(df, window_size):
    # Smooth the data using the moving average filter
    smoothed_df = df.copy()
    for column in df.columns:
        smoothed_df[column] = df[column].rolling(window=window_size, min_periods=1).mean()
    return smoothed_df

def get_data_range(df):
    # Get the range of indices using a slider
    start_index = st.slider("Start Index", 0, len(df) - 1, 0)
    end_index = st.slider("End Index", start_index, len(df) - 1, len(df) - 1)
    return start_index, end_index

def plot_selected_data_with_bounds(df, start_index, end_index, selected_columns):
    # Plot the selected data for selected columns with bounds using matplotlib
    st.subheader("Selected Data Visualization with Bounds:")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for column in selected_columns:
        ax.plot(df.index, df[column], label=None)

    # Highlight the selected range with vertical red lines
    ax.axvline(x=start_index, color='red', linestyle='--', label='Selected Range Start')
    ax.axvline(x=end_index, color='red', linestyle='--', label='Selected Range End')

    ax.set_xlabel("Index")
    ax.set_ylabel("Data")
    st.pyplot(fig)

def plot_selected_data(df, start_index, end_index, selected_columns):
    # Plot the data between the selected vertical lines for selected columns using matplotlib
    st.subheader("Selected Data Visualization between Vertical Lines:")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for column in selected_columns:
        ax.plot(df.index[start_index:end_index + 1], df[column].iloc[start_index:end_index + 1], label=None)

    # Highlight the selected range with vertical red lines
    ax.axvline(x=start_index, color='red', linestyle='--', label='Selected Range Start')
    ax.axvline(x=end_index, color='red', linestyle='--', label='Selected Range End')

    ax.set_xlabel("Index")
    ax.set_ylabel("Data")
    st.pyplot(fig)


def save_selected_data(selected_df):
    # Button to save selected data to file with an option to name the file
    file_name = st.text_input("Enter file name (including extension):", "selected_data.csv")
    if st.button("Save"):
        selected_df.to_csv(file_name, index=False)
        st.success(f"Selected data saved to {file_name}")

if __name__ == "__main__":
    main()