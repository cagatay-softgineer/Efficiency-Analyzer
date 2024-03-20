from Line import Line_Graph_Gen as Ln
from Sunburst import SunBurst_Graph_Gen as SB
from folder_name_gather import change_folder_names_in_html as HTML_Update

date="2024.03.14"
output_path = 1
while output_path < 5:
    
    #print(f"Started : {output_path}")
    csv = f"CSV/Result 2/Cut{output_path}.csv"
    Ln(csv,output_path,date)
    csv = f"CSV/Result 2/Merged.csv"
    SB(csv,output_path,date)
    print(f"Finished : {output_path}")
    output_path += 1
    

HTML_Update("Report/index.html","Report")