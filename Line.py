import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from PDFGen import generate_pdf as PDF_OUT
import statistics
from collections import Counter
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy import stats

def Line_Graph_Gen(csv_path,output_path,date):
    directory_path = os.path.join("Report", date, f"Kesim{output_path}", "Graph")
    
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    
    data = pd.read_csv(csv_path, encoding='ISO-8859-9')

    Timestamp                   = data["Timestamp"]
    Makine_Calisma              = data["Makine_Calisma"]
    Insan_Calisma               = data["Insan_Toplam_Calisma"]
    Tespit_Edilen_Insan_Sayisi  = data["Tespit_Edilen_Insan_Sayisi"]
    Insan_Kirma                 = data["Insan_Kirma"]
    Insan_Tasima                = data["Insan_Tasima"]
    Durum_108                   = data["108_Durumu"]
    Durum_106                   = data["106_Durumu"]
    Insan_Durum                 = data["Insan_Durumu"]


    ##### Array Fonksiyonları ######

    def Select_Mode_or_Median(veri):
        # Mod ve medyanı hesapla
        mod = stats.mode(veri)
        medyan = np.median(veri)

        # Veri setinin dağılımını değerlendir
        veri_std = np.std(veri)

        # Veri setinin dağılımına bağlı olarak karar ver
        if veri_std < 1:
            #print("Veri seti düşük varyansa sahip, mod kullanılabilir.")
            return mod.mode[0]
        else:
            #print("Veri seti yüksek varyansa sahip, medyan kullanılabilir.")
            return medyan
    
    def find_most_consecutive_areas(arr):

        n = len(arr)
        true_areas = []
        false_areas = []

        i = 0
        while i < n:
            # Find consecutive True areas
            while i < n and arr[i]:
                start_true = i
                while i < n and arr[i]:
                    i += 1
                end_true = i - 1
                true_areas.append((start_true, end_true))

            # Find consecutive False areas
            while i < n and not arr[i]:
                start_false = i
                while i < n and not arr[i]:
                    i += 1
                end_false = i - 1
                false_areas.append((start_false, end_false))

        return true_areas, false_areas

    def find_largest_size_and_range(ranges):
        largest_size = 0
        largest_range = None

        for start, end in ranges:
            size = end - start + 1
            if size > largest_size:
                largest_size = size
                largest_range = (start, end)

        return largest_size, largest_range

    def count_true_values(array, count=0):
        count += sum(1 for value in array if value)
        return count

    def calculate_avg(array):
        total = 0
        for value in array:
            total += abs(value[1]-value[0])

        return total/(len(array))   

    def compare_ranges_with_number(ranges, comparison_number):
        result_dict = {'bigger': [], 'smaller': []}

        for start, end in ranges:
            size = end - start + 1

            if size > comparison_number:
                result_dict['bigger'].append((start, end))
            elif size < comparison_number:
                result_dict['smaller'].append((start, end))

        return result_dict    

    def compare_ranges_with_median(ranges):
        sizes = [end - start + 1 for start, end in ranges]
        median_size = statistics.median(sizes)

        result_dict = {'bigger': [], 'smaller': [], 'equal': []}

        for start, end in ranges:
            size = end - start + 1

            if size > median_size:
                result_dict['bigger'].append((start, end))
            elif size < median_size:
                result_dict['smaller'].append((start, end))
            else:
                result_dict['equal'].append((start, end))    

        return result_dict
    
    def get_period_times(Consecutive_True_Array,Consecutive_False_Array):
        len_array = []
        if len(Consecutive_False_Array) == len(Consecutive_True_Array) or True:
            for index in range(len(Consecutive_True_Array)):
                if index < len(Consecutive_True_Array) and index < len(Consecutive_False_Array):
                    len_array.append((abs(Consecutive_True_Array[index][0]-Consecutive_False_Array[index][1])))

            #print(len_array)    
            return len_array

        else:
            print("not Equal")  
            return []

    def get_periods(Consecutive_True_Array,Consecutive_False_Array):
        len_array = []
        if len(Consecutive_False_Array) == len(Consecutive_True_Array) or True:
            for index in range(len(Consecutive_True_Array)):
                if index < len(Consecutive_True_Array) and index < len(Consecutive_False_Array):
                    len_array.append((Consecutive_True_Array[index][0],Consecutive_False_Array[index][1]))

            return len_array

        else:
            print("not Equal")  
            return []

    def group_by_pertanges1(NoN_Group_Array):
        # Group percentages into different ranges (10%, 20%, 30%, 40%, 50%+)
        grouped_counts = Counter()

        for percentage in NoN_Group_Array:
            if percentage < 10:
                grouped_counts['0-10%'] += 1
            elif percentage < 20:
                grouped_counts['10-19%'] += 1
            elif percentage < 30:
                grouped_counts['20-29%'] += 1
            elif percentage < 40:
                grouped_counts['30-39%'] += 1
            elif percentage < 50:
                grouped_counts['40-49%'] += 1
            elif percentage < 60:
                grouped_counts['50-59%'] += 1
            elif percentage < 70:
                grouped_counts['60-69%'] += 1
            elif percentage < 80:
                grouped_counts['70-79%'] += 1
            elif percentage < 90:
                grouped_counts['80-89%'] += 1
            elif percentage < 100:
                grouped_counts['90-99%'] += 1
            else:
                grouped_counts['>=100%'] += 1

        # Define a custom sorting order for the keys
        sorting_order = ['0-10%', '10-19%', '20-29%', '30-39%', '40-49%', '50-59%', '60-69%', '70-79%', '80-89%', '90-99%', '>=100%']

        # Sort the Counter by the custom sorting order
        sorted_counts = Counter({key: grouped_counts[key] for key in sorting_order})
        return sorted_counts

    def group_by_pertanges(NoN_Group_Array):
        # Group percentages into different ranges (10%, 20%, 30%, 40%, 50%+)
        grouped_counts = Counter()

        for percentage in NoN_Group_Array:
            if percentage < 30:
                grouped_counts['0-30%'] += 1
            elif percentage < 60:
                grouped_counts['30-60%'] += 1
            elif percentage < 100:
                grouped_counts['60-100%'] += 1
            else:
                grouped_counts['>=100%'] += 1

        # Define a custom sorting order for the keys
        sorting_order = ['0-30%', '30-60%', '60-100%','>=100%']

        # Sort the Counter by the custom sorting order
        sorted_counts = Counter({key: grouped_counts[key] for key in sorting_order})
        return sorted_counts

    def calculate_avg_for_groups(NoN_Group_Array, grouped_indexes):
        averages = {}
        for percentage_range, indexes in grouped_indexes.items():
            if indexes:
                values_in_group = [NoN_Group_Array[i] for i in indexes]
                average = sum(values_in_group) / len(values_in_group)
                minutes = int(average / 60)
                remaining_seconds = average % 60
                averages[percentage_range] = "{:02d} Dakika {:02d} Saniye".format(minutes, int(remaining_seconds))
            else:
                averages[percentage_range] = percentage_range  # Handle empty groups
        return averages

    def group_by_calculate_Avg(NoN_Group_Array):
        # Group percentages into different ranges (0-30%, 30-60%, 60-100%, >=100%)
        grouped_counts = {'0-30%': [], '30-60%': [], '60-100%': [], '>=100%': []}

        for index, percentage in enumerate(NoN_Group_Array):
            if percentage < 30:
                grouped_counts['0-30%'].append(index)
            elif percentage < 60:
                grouped_counts['30-60%'].append(index)
            elif percentage < 100:
                grouped_counts['60-100%'].append(index)
            else:
                grouped_counts['>=100%'].append(index)

        # Define a custom sorting order for the keys
        sorting_order = ['0-30%', '30-60%', '60-100%', '>=100%']

        # Sort the dictionary by the custom sorting order
        sorted_counts = {key: grouped_counts[key] for key in sorting_order}
        return sorted_counts
    
    
    TimestampMAX = Timestamp.iloc[-1]+1
    Makine_CalismaTOPLAM = count_true_values(Makine_Calisma)   # Makine toplam çalışma süre
    InsanTOP=count_true_values(Insan_Calisma)                  # Insan toplam çalışma süre
    InsanSURE=count_true_values(Tespit_Edilen_Insan_Sayisi)

    #Makine Veriler
    Makine_Consecutive_True     ,   Makine_Consecutive_False   = find_most_consecutive_areas(Makine_Calisma)

    Makine_Largest_True_Size    ,   Makine_Largest_True_Range  = find_largest_size_and_range(Makine_Consecutive_True)
    Makine_Largest_False_Size   ,  Makine_Largest_False_Range  = find_largest_size_and_range(Makine_Consecutive_False)

    #Insan Veriler
    Insan_Consecutive_True      ,  Insan_Consecutive_False      = find_most_consecutive_areas(Insan_Calisma)

    Insan_Largest_True_Size     ,  Insan_Largest_True_Range     = find_largest_size_and_range(Insan_Consecutive_True)
    Insan_Largest_False_Size    ,  Insan_Largest_False_Range    = find_largest_size_and_range(Insan_Consecutive_False)

    #print(Makine_Consecutive_True)
    #print()
    #print(Makine_Consecutive_False)
    #print()
    #print()
    #print(len(Makine_Consecutive_True))
    #print()
    #print(len(Makine_Consecutive_False))
    #print()
    #print()

    Makine_Calisma_Periods = get_period_times(Makine_Consecutive_True,Makine_Consecutive_False)

    Makine_Calisma_Periods_By = get_periods(Makine_Consecutive_True,Makine_Consecutive_False)


    Makine_Calisma_Periods_lenghts = [b - a for a, b in Makine_Calisma_Periods_By]
    
    Makine_Calisma_Periods_lenghts_np = np.array(Makine_Calisma_Periods_lenghts)
    
    Makine_Calisma_Periods_lenghts_sorted = sorted(Makine_Calisma_Periods_lenghts)

    Makine_Calisma_Periods_lenghts_Median = statistics.median(sorted(Makine_Calisma_Periods_lenghts))
    
    Makine_Calisma_Periods_lenghts_Mode = Counter(sorted(Makine_Calisma_Periods_lenghts)).most_common(1)[0][0]

    #print(Makine_Calisma_Periods_By)
    #print()
    # Find the minimum and maximum lengths
    min_length = min(Makine_Calisma_Periods_lenghts)
    max_length = max(Makine_Calisma_Periods_lenghts)

    # Calculate cumulative sum of lengths
    cumulative_sum = [sum(Makine_Calisma_Periods_lenghts[:i+1]) for i in range(len(Makine_Calisma_Periods_lenghts))]

    # Find the index where cumulative sum exceeds half of the total sum of lengths
    total_sum = sum(Makine_Calisma_Periods_lenghts)
    half_sum = total_sum / 2
    avg_index = next(i for i, val in enumerate(cumulative_sum) if val >= half_sum)

    # Define threshold
    threshold = 500

    # Find all common average indexes within the threshold
    common_avg_indexes = [i for i, val in enumerate(cumulative_sum) if abs(val - cumulative_sum[avg_index]) <= threshold]

    Makine_Calisma_common_avg_periods = [Makine_Calisma_Periods_By[i] for i in common_avg_indexes]

    #print(Makine_Calisma_common_avg_periods)

    #print(avg_index)

    # Find the indexes of the minimum and maximum lengths in the lengths array
    min_index = Makine_Calisma_Periods_lenghts.index(min_length)
    max_index = Makine_Calisma_Periods_lenghts.index(max_length)

    # Find the tuple with the minimum second element
    Makine_Calisma_min_period = Makine_Calisma_Periods_By[min_index]

    # Find the tuple with the maximum second element
    Makine_Calisma_max_period = Makine_Calisma_Periods_By[max_index]

    Makine_Calisma_avg_period = Makine_Calisma_Periods_By[avg_index]
    
    Makine_Calisma_Min_Period_Time = min(Makine_Calisma_Periods)
    
    #print()
    #print(Makine_Calisma_Min_Period_Time)
    #print()
    Makine_Calisma_Uneccesary_Wait_Time = sum([abs(percentage-Makine_Calisma_Min_Period_Time) for percentage in Makine_Calisma_Periods])
    #print(Makine_Calisma_Periods)
    #print()
    #print(Makine_Calisma_Uneccesary_Wait_Time)
    #print()

    # Calculate the percentage increase for each element
    Makine_Calisma_Periodical_Work_Ratio = [(value - Makine_Calisma_Min_Period_Time) / Makine_Calisma_Min_Period_Time * 100 for value in Makine_Calisma_Periods]

    # Format each percentage to display only two decimal places
    Makine_Calisma_Periodical_Work_Ratio_Perctanges = ["{:.2f}%".format(percentage) for percentage in Makine_Calisma_Periodical_Work_Ratio]
    #print()
    #print(Makine_Calisma_Periodical_Work_Ratio_Perctanges)
    #print()
    #print(Makine_Calisma_Periods_lenghts)
    #print()
    #print(sorted(Makine_Calisma_Periods_lenghts))
    #print()
    #print(statistics.median(sorted(Makine_Calisma_Periods_lenghts)))
    #print()
    #print(Counter(sorted(Makine_Calisma_Periods_lenghts)).most_common(1)[0][0])
    #print()

    Makine_Calisma_Periodical_Work_Ratio_Group = group_by_pertanges(Makine_Calisma_Periodical_Work_Ratio)
    Makine_Calisma_Periodical_Work_Ratio_Group_indexes = group_by_calculate_Avg(Makine_Calisma_Periodical_Work_Ratio)
    Makine_Calisma_Periodical_Work_Ratios_Avg = calculate_avg_for_groups(Makine_Calisma_Periods_lenghts,Makine_Calisma_Periodical_Work_Ratio_Group_indexes)

    #print(Makine_Calisma_Periodical_Work_Ratios_Avg)
    
    #for key, value in Makine_Calisma_Periodical_Work_Ratio_Group.items():
    #    print(f"{key}: {value}")


    time_values = [pd.to_datetime(sec, unit='s').strftime('%H:%M:%S') for sec in Timestamp]

    # Sol Graph
    fig = px.line()
    config = {
      'toImageButtonOptions': {
        'format': 'png', # one of png, svg, jpeg, webp
        'filename': 'Grafik',
        'height': 3840,
        'width': 2560,
        'scale': 1, # Multiply title/legend/axis/canvas sizes by this factor
      },'displaylogo': False,
    }

    #Makine Çalışma Verisi
    fig.add_scatter(x=time_values, y=Makine_Calisma.astype(int), 
                    mode='lines', name='Makine Çalışma Verisi', 
                    line=dict(width=5,color='rgba(65,105,225,1)'), marker=dict(color='orange'), 
                    showlegend=True, legendgroup="makine", hovertemplate = '<b>Makine : </b>%{y}<extra></extra>')

    fig.add_vrect(
    x0=Makine_Calisma_min_period[0], x1=Makine_Calisma_min_period[1],
    fillcolor="orchid", opacity=0.5, 
    layer="above", line_width = 1,legendgroup="makine",showlegend=True,name="Periodu En Kısa Aralık",
    label=dict(text="Min",textangle=90,font=dict(size=20, family="arial"),),
    )
    fig.add_vrect(
    x0=Makine_Calisma_max_period[0], x1=Makine_Calisma_max_period[1],
    fillcolor="seagreen", opacity=0.5,
    layer="above", line_width=1,legendgroup="makine",showlegend=True,name="Periodu En Uzun Aralık",
    label=dict(text="Max",textangle=90,font=dict(size=20, family="arial")),
    )

    fig.update_traces(line=dict(shape='hvh'))

    fig.update_layout(
        title='<b>Makine Çalışma Grafiği</b>',
        xaxis_title='<b>Zaman</b>',
        yaxis_title='<b>Durum</b>',
        xaxis=dict(showgrid=False, gridwidth=1, gridcolor='darkgrey',tickfont=dict(size=12), showspikes=True, spikethickness=2, spikedash="dash",spikesnap="cursor", spikecolor="rgba(153, 153, 153,0.8)", spikemode="across",rangeslider=dict(visible=False),automargin=True,exponentformat='none'),  # Izgara özellikleri
        yaxis=dict(showgrid=False, gridwidth=1, gridcolor='darkgrey',tickfont=dict(size=12), tickvals=[1,0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], ticktext=['<b>Makine Çalışıyor</b>', '<b>Makine Çalışmıyor</b>',0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        legend=dict(title='<b>Veri Setleri</b>', yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h",entrywidth=200,font=dict(family="Courier",size=12,color="black"),bgcolor="white",bordercolor="Black",borderwidth=2),
        plot_bgcolor='white',  # Grafik arka plan rengi
        hovermode="x",
        hoverdistance=100, # Distance to show hover label of data point
        spikedistance=1000, # Distance to show spike
        xaxis_tickformatstops = [
            dict(dtickrange=["min","max"], value="<b>%H:%M:%S s</b>"),],
        autosize=True,
        margin=dict(
            l=100,
            r=100,
            b=100,
            t=100,
            pad=4
        ),
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(ticklabelstep=3,tickangle=90)
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    
    # Sağ Graph
    fig2 = px.line()
    fig2.add_scatter(x=time_values, y=Insan_Durum, mode='lines', name='İnsan', line=dict(width=2), marker=dict(color='midnightblue'), offsetgroup="insan", hovertemplate = '<b>Durum :</b> %{y}<extra></extra>',showlegend=True)
    #fig2.add_scatter(x=time_values, y=Durum_106, mode='lines', name='106', line=dict(width=2), marker=dict(color='blue'), offsetgroup="insan", hovertemplate = '106 Durumu : %{y}<extra></extra>',showlegend=True)
    fig2.add_shape(
        legendrank=2,
        showlegend=False,
        type="line",
        xref="paper",
        line=dict(dash="15px"),
        x0=0.00,
        x1=1.0,
        y0=0,
        y1=0,
    )
    fig2.update_traces(line=dict(shape='hvh'))

    fig2.update_layout(
        title='<b>İnsan Çalışma Grafiği</b>',
        xaxis_title='<b>Zaman</b>',
        yaxis_title='<b>Durum</b>',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='darkgrey',tickfont=dict(size=12), showspikes=True, spikethickness=2, spikedash="dot", spikecolor="#999999", spikemode="across",rangeslider=dict(visible=False),automargin=True,exponentformat='none'),  # Izgara özellikleri
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='darkgrey',tickfont=dict(size=12),tickvals=[-1,0,1,2], ticktext=['<b>Çalışma Alanında Kimse Yok</b>','<b>İnsanlar Çalışmıyor</b>','<b>Bir Kişi Çalışıyor</b>','<b>İnsanlar Çalışıyor</b>']),
        legend=dict(title='<b>Veri Setleri</b>', yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h",entrywidth=200,font=dict(family="Courier",size=12,color="black"),bgcolor="white",bordercolor="Black",borderwidth=2),
        plot_bgcolor='white',  # Grafik arka plan rengi
        hovermode="x",
        hoverdistance=100, # Distance to show hover label of data point
        spikedistance=1000, # Distance to show spike
        xaxis_tickformatstops = [
            dict(dtickrange=["min","max"], value="<b>%H:%M:%S s</b>")],
        autosize=True,    
        margin=dict(
            l=100,
            r=100,
            b=100,
            t=100,
            pad=4
        ),
    )
    fig2.update_yaxes(automargin=True)

    makine_hareket_toplam_calisma = len([i for i, value in enumerate(Makine_Calisma) if value])

    makine_hareket_toplam_calisma_HMS_formatted = datetime.now().replace(
                hour=int(makine_hareket_toplam_calisma // 3600),
            minute=int((makine_hareket_toplam_calisma % 3600) // 60),
            second=int(makine_hareket_toplam_calisma % 60)
        ).strftime("%H:%M:%S")

    makine_hareket_text = f'Makine Hareketi Toplam Süre : {makine_hareket_toplam_calisma_HMS_formatted}'


    toplam_calisma_süresi_HMS_formatted = datetime.now().replace(
                            hour=int(max(len(Insan_Calisma),len(Makine_Calisma)) // 3600),
                        minute=int((max(len(Insan_Calisma),len(Makine_Calisma)) % 3600) // 60),
                        second=int(max(len(Insan_Calisma),len(Makine_Calisma)) % 60)
                    ).strftime("%H:%M:%S")

    toplam_calisma_süresi_text = f'Video Toplam Süre: {toplam_calisma_süresi_HMS_formatted}'


    insan_hareket_calisan0_toplam_calisma_süresi = len([i for i, value in enumerate(Insan_Kirma) if value])
    insan_hareket_calisan0_toplam_calisma_süresi_HMS_formatted = datetime.now().replace(
                hour=int(insan_hareket_calisan0_toplam_calisma_süresi // 3600),
            minute=int((insan_hareket_calisan0_toplam_calisma_süresi % 3600) // 60),
            second=int(insan_hareket_calisan0_toplam_calisma_süresi % 60)
        ).strftime("%H:%M:%S")

    insan_hareket_calisan0_toplam_calisma_süresi_text = f'Çalışan-0 Çalışma zamanı {insan_hareket_calisan0_toplam_calisma_süresi_HMS_formatted}'


    insan_hareket_calisan1_toplam_calisma_süresi = len([i for i, value in enumerate(Insan_Tasima) if value])
    insan_hareket_calisan1_toplam_calisma_süresi_HMS_formatted = datetime.now().replace(
                hour=int(insan_hareket_calisan1_toplam_calisma_süresi // 3600),
            minute=int((insan_hareket_calisan1_toplam_calisma_süresi % 3600) // 60),
            second=int(insan_hareket_calisan1_toplam_calisma_süresi % 60)
        ).strftime("%H:%M:%S")

    insan_hareket_calisan1_toplam_calisma_süresi_text = f'Çalışan-1 Çalışma zamanı {insan_hareket_calisan1_toplam_calisma_süresi_HMS_formatted}'


    annotation_makine_hareket = dict(x=0.10, y=-0.50,
                       text=makine_hareket_text,
                       showarrow=False, font=dict(color='green', size=12), align='left', xref='paper', yref='paper')

    annotation_toplam_calisma_süresi = dict(x=0.90, y=-0.50,
                       text=toplam_calisma_süresi_text,
                       showarrow=False, font=dict(color='black', size=12), align='left', xref='paper', yref='paper')

    annotation_insan_hareket_calisan0_toplam_calisma_süresi = dict(x=0.70, y=-0.20,
                       text=insan_hareket_calisan0_toplam_calisma_süresi_text,
                       showarrow=False, font=dict(color='red', size=12), align='left', xref='paper', yref='paper')

    annotation_insan_hareket_calisan1_toplam_calisma_süresi = dict(x=0.90, y=-0.20,
                       text=insan_hareket_calisan1_toplam_calisma_süresi_text,
                       showarrow=False, font=dict(color='blue', size=12), align='left', xref='paper', yref='paper')

    #fig.update_layout(annotations=[annotation_makine_hareket,annotation_toplam_calisma_süresi])

    fig.update_traces(marker=dict(size=3))

    fig2.update_traces(marker=dict(size=3))

    fig2.update_layout(xaxis_tickformatstops = [
            dict(dtickrange=[None,None], value="<b>%H:%M:%S s</b>")],
                       )
    fig2.update_xaxes(ticklabelstep=3,tickangle=90,ticklen=10)

    fig2.update_layout(legend=dict(groupclick="toggleitem"))

    custom_colors = {
        '0-10%': '#00cc00',
        '10-19%': '#33cc00',
        '20-29%': '#66cc00',
        '30-39%': '#99cc00',
        '40-49%': '#cccc00',
        '50-59%': '#cc9900',
        '60-69%': '#cc6600',
        '70-79%': '#cc3300',
        '80-89%': '#cc0000',
        '90-99%': '#cc0033',
        '>=100%': '#cc0066'
    }
    custom_colors = {
        list(Makine_Calisma_Periodical_Work_Ratios_Avg.keys())[0] : '#00cc00',
        list(Makine_Calisma_Periodical_Work_Ratios_Avg.keys())[1] : '#cccc00',
        list(Makine_Calisma_Periodical_Work_Ratios_Avg.keys())[2] : '#cc3300',
        list(Makine_Calisma_Periodical_Work_Ratios_Avg.keys())[3] : '#000000',
    }
    performance_text = ['High Performans', 'Medium Performans', 'Low Performans', 'Criticaly Low Performans']
    # Create a bar chart using plotly express
    fig3 = px.bar(x=list(Makine_Calisma_Periodical_Work_Ratios_Avg.values()),
                  y=list(Makine_Calisma_Periodical_Work_Ratio_Group.values()),
                  labels={'x': 'Aralık : ', 'y': 'Sayı : '},
                  title='Makine Calisma Periodical Work Ratio Group',
                  color=list(Makine_Calisma_Periodical_Work_Ratio_Group.keys()),
                  color_discrete_map=custom_colors,
                  #text=list(Makine_Calisma_Periodical_Work_Ratios_Avg.values()),
                  text=[f'{y_val} : {performance_text[i]}' for i, y_val in enumerate(list(Makine_Calisma_Periodical_Work_Ratio_Group.values()))],
                  #text_auto=''
                  )

    fig3.update_layout(
        title=f'<b>Personel Cam Kırım Performans Grafiği           Toplam Plate Sayısı :</b> {sum(Makine_Calisma_Periodical_Work_Ratio_Group.values())}',
        xaxis_title='<b>Performans Dağılımı</b>',
        yaxis_title='<b>Cam Kesim Sayısı</b>',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='darkgray',tickfont=dict(size=16), showspikes=True, spikethickness=2, spikedash="dash",spikesnap="cursor", spikecolor="rgba(153, 153, 153,0.8)", spikemode="across",rangeslider=dict(visible=False),automargin=True,exponentformat='none'),  # Izgara özellikleri
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='darkgray',tickfont=dict(size=16),tickmode='linear', tick0=0),
        legend=dict(title='<b>Veri Setleri</b>', yanchor="bottom", y=1.02, xanchor="right", x=1, orientation="h",entrywidth=50,font=dict(family="Courier",size=12,color="black"),bgcolor="white",bordercolor="Black",borderwidth=2),
        plot_bgcolor='white',  # Grafik arka plan rengi
        hovermode="x",
        hoverdistance=100, # Distance to show hover label of data point
        spikedistance=1000, # Distance to show spike
        xaxis_tickformatstops = [
            dict(dtickrange=["min","max"], value="<b>%H:%M:%S s</b>"),],
        autosize=True,
        bargap=0.5,
        font_color="black",
        font=dict(size=16),
        margin=dict(
            l=100,
            r=100,
            b=100,
            t=100,
            pad=4
        ),
    )
    fig3.update_yaxes(automargin=True)
    fig3.update_xaxes(tickangle=0)
    hover_template = '<b>Ortalama Süre : </b> %{x}<br><b>Kesilen Cam Sayısı : </b> %{y}'
    fig3.update_traces(hovertemplate=hover_template)
    fig3.update_layout(legend=dict(groupclick="toggleitem"))
    fig3.update_traces(textposition='outside')


    fig.write_html(f"Report/{date}/Kesim{output_path}/Graph/Makine-Grafiği.html",config=config)
    fig2.write_html(f"Report/{date}/Kesim{output_path}/Graph/Merged-Data-Right-Grafiği.html",config=config)
    fig3.write_html(f"Report/{date}/Kesim{output_path}/Graph/Makine-Bar-Grafiği.html",config=config)
    # Manually change the page title in the HTML file
    with open(f"Report/{date}/Kesim{output_path}/Graph/Makine-Grafiği.html", 'r', encoding='utf-8') as file:
        html_content = file.read()

        button = '<button class="show-details-btn" onclick="toggleDetails(this)">Anasayfa</button>'
        button2= '<button id="pdfButton" onclick="showReport()">Rapor Göster</button>'
        #btn_script = '<script>function openWebsite() {window.location.href = "Al/0224/same_time/3/output/Merged-Data-New-Normalizer.html";}</script>'
        # Replace the existing title or add it if not present
        new_title = '<title>AISOFT</title>'
        style = '<style>body {background-color: #ffffff;}#pdfButton {position: fixed;bottom: 10px;left: 10px;padding: 10px;background-color: #007bff;color: #fff;border: none;cursor: pointer;}.container {display: flex;}.embed-container {position: relative;width: 50%;height: 30vw;box-sizing: border-box;margin: 0;overflow: hidden;position: relative;}.embed-container embed {width: 100%;height: 100%;}.show-details-btn {position: absolute;top: 10px;left: 10px;background-color: #1cd422;color: white;padding: 5px 10px;font-size: 16px;border: none;border-radius: 5px;cursor: pointer;}</style>'
        # Add or replace the favicon link
        favicon_link = '<link rel="icon" href="https://www.aisofttechnology.com/wp-content/uploads/2023/11/cropped-aisoft_fav-192x192.png" type="image/x-icon">'
        script = '<script>function toggleDetails(button) {window.history.back();}</script>'
        script2 = '<script>function showReport() {window.open("{{ url_for("ReportPDF") }}", "_blank");}</script>'
        html_content = html_content.replace('</head>',f'{new_title}\n {style}\n {favicon_link}\n</head>')
        #html_content = html_content.replace('</body>', f'{button}\n {script}\n {button2}\n {script2}\n </body>')
        html_content = html_content.replace("Durum Zaman Grafiği", " ")

    with open(f"Report/{date}/Kesim{output_path}/Graph/Makine-Grafiği.html", 'w', encoding='utf-8') as file:
        file.write(html_content)

    with open(f"Report/{date}/Kesim{output_path}/Graph/Merged-Data-Right-Grafiği.html", 'r', encoding='utf-8') as file:
        html_content = file.read()
        button = '<button class="show-details-btn" onclick="toggleDetails(this)">Anasayfa</button>'
        #btn_script = '<script>function openWebsite() {window.location.href = "Al/0224/same_time/3/output/Merged-Data-New-Normalizer.html";}</script>'
        # Replace the existing title or add it if not present
        new_title = '<title>AISOFT</title>'
        style = '<style>body {background-color: #ffffff;}.container {display: flex;}.embed-container {position: relative;width: 50%;height: 30vw;box-sizing: border-box;margin: 0;overflow: hidden;position: relative;}.embed-container embed {width: 100%;height: 100%;}.show-details-btn {position: absolute;top: 10px;left: 10px;background-color: #1cd422;color: white;padding: 5px 10px;font-size: 16px;border: none;border-radius: 5px;cursor: pointer;}</style>'
        # Add or replace the favicon link
        favicon_link = '<link rel="icon" href="https://www.aisofttechnology.com/wp-content/uploads/2023/11/cropped-aisoft_fav-192x192.png" type="image/x-icon">'
        script = '<script>function toggleDetails(button) {window.history.back();}</script>'
        html_content = html_content.replace('</head>',f'{new_title}\n {style}\n {favicon_link}\n</head>')
        #html_content = html_content.replace('</body>', f'{button}\n {script}\n</body>')
        html_content = html_content.replace("Durum Zaman Grafiği", " ")
    with open(f"Report/{date}/Kesim{output_path}/Graph/Merged-Data-Right-Grafiği.html", 'w', encoding='utf-8') as file:
        file.write(html_content)

    with open(f"Report/{date}/Kesim{output_path}/Graph/Makine-Bar-Grafiği.html", 'r', encoding='utf-8') as file:
        html_content = file.read()
        button = '<button class="show-details-btn" onclick="toggleDetails(this)">Anasayfa</button>'
        # Replace the existing title or add it if not present
        new_title = '<title>AISOFT</title>'
        style = '<style>body {background-color: #ffffff;}.container {display: flex;}.embed-container {position: relative;width: 50%;height: 30vw;box-sizing: border-box;margin: 0;overflow: hidden;position: relative;}.embed-container embed {width: 100%;height: 100%;}.show-details-btn {position: absolute;top: 10px;left: 10px;background-color: #1cd422;color: white;padding: 5px 10px;font-size: 16px;border: none;border-radius: 5px;cursor: pointer;}</style>'
        # Add or replace the favicon link
        favicon_link = '<link rel="icon" href="https://www.aisofttechnology.com/wp-content/uploads/2023/11/cropped-aisoft_fav-192x192.png" type="image/x-icon">'
        script = '<script>function toggleDetails(button) {window.history.back();}</script>'
        html_content = html_content.replace('</head>',f'{new_title}\n {style}\n {favicon_link}\n</head>')
        #html_content = html_content.replace('</body>', f'{button}\n {script}\n</body>')
        html_content = html_content.replace("Durum Zaman Grafiği", " ")
    with open(f"Report/{date}/Kesim{output_path}/Graph/Makine-Bar-Grafiği.html", 'w', encoding='utf-8') as file:
        file.write(html_content)

    # Initialize output_arr
    output_arr1 = []
    output_arr2 = []
    # Append formatted strings to output_arr for Makine and Insan
    for value in Makine_Consecutive_False:
        start_time_value = datetime.now().replace(
            hour=int(value[0] // 3600),
            minute=int((value[0] % 3600) // 60),
            second=int(value[0] % 60)
        ).strftime("%H:%M:%S")

        stop_time_value = datetime.now().replace(
            hour=int(value[1] // 3600),
            minute=int((value[1] % 3600) // 60),
            second=int(value[1] % 60)
        ).strftime("%H:%M:%S")

        output_arr1.append(f"{start_time_value} - {stop_time_value} : Makine Çalışmıyor!")

    for value in Insan_Consecutive_False:
        start_time_value = datetime.now().replace(
            hour=int(value[0] // 3600),
            minute=int((value[0] % 3600) // 60),
            second=int(value[0] % 60)
        ).strftime("%H:%M:%S")

        stop_time_value = datetime.now().replace(
            hour=int(value[1] // 3600),
            minute=int((value[1] % 3600) // 60),
            second=int(value[1] % 60)
        ).strftime("%H:%M:%S")

        output_arr2.append(f"{start_time_value} - {stop_time_value} : İnsan Çalışmıyor!")
        #output_arr1.append(f"{start_time_value} - {stop_time_value} : Makine Çalışmıyor!")

    #PDF_OUT(output_arr1,output_arr2,"Report/PDF/Rapor.pdf")