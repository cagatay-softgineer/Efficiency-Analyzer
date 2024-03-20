import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import os 

config = {
  'toImageButtonOptions': {
    'format': 'png', # one of png, svg, jpeg, webp
    'filename': 'Grafik',
    'height': 3840,
    'width': 2560,
    'scale': 1, # Multiply title/legend/axis/canvas sizes by this factor
  },'displaylogo': False,
}
def count_true_values(array, count=0):
    count += sum(1 for value in array if value)
    return count

def count_values(array, count=0):
    count += sum(1 for value in array)
    return count

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
    
def count_each(arr):
    # Initialize counts
    count_2 = count_1 = count_0 = count_minus_1 = 0
    
    # Count occurrences
    for value in arr:
        if value == 2:
            count_2 += 1
        elif value == 1:
            count_1 += 1
        elif value == 0:
            count_0 += 1
        elif value == -1:
            count_minus_1 += 1
    return count_2, count_1, count_0, count_minus_1


csv = "CSV/Cut4.csv"

def SunBurst_Graph_Gen(csv,output_path,date):
        
    # CSV dosyalarını oku
    veri = pd.read_csv(csv, encoding='ISO-8859-9')

    #toplam_süre = veri['Timestamp'].to_list()[-1:][0]
    #makine_toplam = veri['Makine_Calisma'].to_list()[-1:][0]
    #insan_toplam = veri['Insan_Toplam_Calisma'].to_list()[-1:][0]
    #tespit_toplam = veri['Tespit_Edilen_Insan_Sayisi'].to_list()[-1:][0]

    Timestamp                   = veri["Timestamp"]
    Makine_Calisma              = veri["Makine_Calisma"]
    Insan_Calisma               = veri["Insan_Toplam_Calisma"]
    Tespit_Edilen_Insan_Sayisi  = veri["Tespit_Edilen_Insan_Sayisi"]
    Insan_Kirma                 = veri["Insan_Kirma"]
    Insan_Tasima                = veri["Insan_Tasima"]
    Insan_Durum                 = veri["Insan_Durumu"]

    toplam_süre = Timestamp.iloc[-1]+1
    makine_toplam = count_values(Makine_Calisma)   # Makine toplam çalışma süre
    insan_toplam = count_true_values(Insan_Calisma)           # Insan toplam çalışma süre
    tespit_toplam = count_true_values(Tespit_Edilen_Insan_Sayisi)

    count_2, count_1, count_0, count_minus_1 = count_each(Insan_Durum)




    #Makine Veriler
    Makine_Consecutive_True     ,   Makine_Consecutive_False   = find_most_consecutive_areas(Makine_Calisma)

    Makine_Largest_True_Size    ,   Makine_Largest_True_Range  = find_largest_size_and_range(Makine_Consecutive_True)
    Makine_Largest_False_Size   ,  Makine_Largest_False_Range  = find_largest_size_and_range(Makine_Consecutive_False)

    Makine_Calisma_Periods = get_period_times(Makine_Consecutive_True,Makine_Consecutive_False)

    # Find the tuple with the minimum second element
    Makine_Calisma_Min_Period_Time = min(Makine_Calisma_Periods)


    Makine_Calisma_Uneccesary_Wait_Time = sum([abs(percentage-Makine_Calisma_Min_Period_Time) for percentage in Makine_Calisma_Periods])

    makine_toplam -= Makine_Calisma_Uneccesary_Wait_Time

    toplam_oran = (toplam_süre/toplam_süre*100)
    makine_calisma_oranı = int(int(makine_toplam)/toplam_süre*100)

    insan_iki_kisi_calisma_oranı = int(int(count_2)/toplam_süre*100)
    insan_bir_kisi_calisma_oranı = int(int(count_1)/toplam_süre*100)
    insan_sifir_kisi_calisma_oranı = int(int(count_0)/toplam_süre*100)
    insan_tespit_edilmeme_oranı = int(int(count_minus_1)/toplam_süre*100)

    # Adjust percentages to ensure the total is 100
    remaining_percentage = 100 - (insan_iki_kisi_calisma_oranı + insan_bir_kisi_calisma_oranı + insan_sifir_kisi_calisma_oranı + insan_tespit_edilmeme_oranı)
    insan_iki_kisi_calisma_oranı += remaining_percentage
    #print(remaining_percentage)

    tespit_oranı =int(int(tespit_toplam)/toplam_süre*100)

    kayıp_oranı = 100 - tespit_oranı

    #print(insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı+insan_sifir_kisi_calisma_oranı+insan_tespit_edilmeme_oranı)

    data = dict(
        character=[f"Makine", f"İnsan", 
                   f"İki Kişi Çalışma<br>{insan_iki_kisi_calisma_oranı}%", 
                   f"Bir Kişi Çalışma<br>{insan_bir_kisi_calisma_oranı}%", 
                   f"Çalışan Yok<br>{insan_sifir_kisi_calisma_oranı}%", 
                   f"Çalışma Alanında<br> Kimse Yok<br>{insan_tespit_edilmeme_oranı}%", 
                   f"Makine Çalışıyor<br>{makine_calisma_oranı}%", 
                   f"Makine Sebepsiz Duruyor<br>{100-makine_calisma_oranı}%",
                   "Analiz"],
        parent=["Analiz", "Analiz", "İnsan", "İnsan", "İnsan", "İnsan", "Makine", "Makine",""], 
        value=[100, 100, insan_iki_kisi_calisma_oranı, insan_bir_kisi_calisma_oranı, insan_sifir_kisi_calisma_oranı, insan_tespit_edilmeme_oranı, makine_calisma_oranı,100-makine_calisma_oranı,toplam_oran*2],
        color=[-50,-50, insan_iki_kisi_calisma_oranı, insan_bir_kisi_calisma_oranı, insan_sifir_kisi_calisma_oranı, insan_tespit_edilmeme_oranı,makine_calisma_oranı, (100-makine_calisma_oranı),-100],
        )

    fig = px.sunburst(
        data,
        names='character',
        parents='parent',
        values='value',
        hover_data=['value'],
        #color='color',
        labels= ["0","0", "50", "30", "20","50", "50","0"],
        branchvalues="total",
        color_continuous_scale=["white","black", "red", "rgb(175, 155, 50)","green"],
        color_continuous_midpoint=0,
        )
    fig.update_traces(hovertemplate='%{label}',)
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0),font=dict(size=28))


    data2 = dict(
        character=[f"Makine Verimi<br>{makine_calisma_oranı}%", 
                   f"İnsan Verimi<br>{insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı}%",
                   f"Toplam Verimlilik<br>{(makine_calisma_oranı+insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı)/2}%"],

        parent=[f"Toplam Verimlilik<br>{(makine_calisma_oranı+insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı)/2}%", 
                f"Toplam Verimlilik<br>{(makine_calisma_oranı+insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı)/2}%",""],
        value=[makine_calisma_oranı,insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı,0],
        color=[makine_calisma_oranı,insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı,(makine_calisma_oranı+insan_iki_kisi_calisma_oranı+insan_bir_kisi_calisma_oranı)/2],
        )


    fig2 = px.sunburst(
        data2,
        names='character',
        parents='parent',
        values='value',
        hover_data=['value'],
        #color='color',
        labels= ["0","0", "50"],
        branchvalues="remainder",
        color_continuous_scale=["white","white","white","black", "red","red", "rgb(175, 155, 50)","rgb(155, 175, 50)","green"],
        color_continuous_midpoint=0,
        )
    fig2.update_traces(hovertemplate='%{label}',)
    fig2.update_layout(margin=dict(t=0, l=0, r=0, b=0),font=dict(size=20))
    fig2.update_xaxes(ticklabelstep=3,tickangle=90)
    fig2.update_xaxes(ticklabelstep=3,tickangle=90)
    fig.write_html(f"Report/{date}/Kesim{output_path}/Graph/Çalışma-Analiz.html",config=config)
    fig2.write_html(f"Report/{date}/Kesim{output_path}/Graph/Verimlilik-Analiz.html",config=config)

    # Manually change the page title in the HTML file
    with open(f"Report/{date}/Kesim{output_path}/Graph/Çalışma-Analiz.html", 'r', encoding='utf-8') as file:
        html_content = file.read()
        new_title = '<title>AISOFT</title>'
        style = '<style>body {background-color: #ffffff;}.container {display: flex;}.embed-container {position: relative;width: 50%;height: 30vw;box-sizing: border-box;margin: 0;overflow: hidden;position: relative;}.embed-container embed {width: 100%;height: 100%;}.show-details-btn {position: absolute;top: 10px;left: 10px;background-color: #1cd422;color: white;padding: 5px 10px;font-size: 16px;border: none;border-radius: 5px;cursor: pointer;}</style>'
        favicon_link = '<link rel="icon" href="https://www.aisofttechnology.com/wp-content/uploads/2023/11/cropped-aisoft_fav-192x192.png" type="image/x-icon">'
        script = '<script>function toggleDetails(button) {window.history.back();}</script>'
        html_content = html_content.replace('</head>',f'{new_title}\n {style}\n {favicon_link}\n</head>')
        html_content = html_content.replace("Durum Zaman Grafiği", " ")

    with open(f"Report/{date}/Kesim{output_path}/Graph/Çalışma-Analiz.html", 'w', encoding='utf-8') as file:
        file.write(html_content)

    with open(f"Report/{date}/Kesim{output_path}/Graph/Verimlilik-Analiz.html", 'r', encoding='utf-8') as file:
        html_content = file.read()
        new_title = '<title>AISOFT</title>'
        style = '<style>body {background-color: #ffffff;}.container {display: flex;}.embed-container {position: relative;width: 50%;height: 30vw;box-sizing: border-box;margin: 0;overflow: hidden;position: relative;}.embed-container embed {width: 100%;height: 100%;}.show-details-btn {position: absolute;top: 10px;left: 10px;background-color: #1cd422;color: white;padding: 5px 10px;font-size: 16px;border: none;border-radius: 5px;cursor: pointer;}</style>'
        favicon_link = '<link rel="icon" href="https://www.aisofttechnology.com/wp-content/uploads/2023/11/cropped-aisoft_fav-192x192.png" type="image/x-icon">'
        script = '<script>function toggleDetails(button) {window.history.back();}</script>'
        html_content = html_content.replace('</head>',f'{new_title}\n {style}\n {favicon_link}\n</head>')
        html_content = html_content.replace("Durum Zaman Grafiği", " ")
    with open(f"Report/{date}/Kesim{output_path}/Graph/Verimlilik-Analiz.html", 'w', encoding='utf-8') as file:
        file.write(html_content)