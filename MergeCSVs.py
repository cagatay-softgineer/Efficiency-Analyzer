import pandas as pd
import csv

def turn_false_if_true_on_both_sides_with_threshold(arr, threshold=1):
    result = arr.copy()

    for i in range(len(arr)):
        if not arr[i]:
            left_side_has_true = any(arr[max(0, max(0,i-1) - threshold-1):max(0,i-1)])
            right_side_has_true = any(arr[i+1:i + threshold + 1])
            
            if left_side_has_true and right_side_has_true:
                result[i] = True

    return result

def turn_true_if_false_on_both_sides_with_threshold(arr, threshold=1):
    result = arr.copy()

    for i in range(len(arr)):
        if arr[i]:
            left_side_has_true = any(arr[max(0, max(0,i-1) - threshold-1):max(0,i-1)])
            right_side_has_true = any(arr[i+1:i + threshold + 1])
            
            if not left_side_has_true and not right_side_has_true:
                result[i] = False

    return result

def smoothe(array,smoothe_threshold,repeat=1,minimal_threshold=1):
    """
    Smoothens the given array by iteratively applying a smoothing operation.

    Parameters:
    - array (list): The input array to be smoothed.
    - smoothe_threshold (int): The threshold for smoothing operation.
    - repeat (int): The number of times to repeat the smoothing process.

    Returns:
    - list: The smoothed array after applying the smoothing process.

    The function iteratively applies a smoothing operation to the given array by turning 
    false elements to true if they have true elements on both sides with values above 
    the specified threshold. This process is repeated 'repeat' times or until no further 
    changes occur in the array.
    """

    for i in range(repeat):

        modified_array = turn_false_if_true_on_both_sides_with_threshold(array, smoothe_threshold)

        if not any_change(array, modified_array): 
            break

        array = turn_true_if_false_on_both_sides_with_threshold(modified_array,minimal_threshold)
    
    return turn_true_if_false_on_both_sides_with_threshold(array,minimal_threshold)

def any_change(arr1, arr2):
    return any(x != y for x, y in zip(arr1, arr2))

def count_worker(arr1, arr2):
    result = []

    for i in range(min(len(arr1), len(arr2))):
        result.append(((1 if arr1[i] else 0) + (1 if arr2[i] else 0)))

    return result

def merge_work_times(arr1, arr2):
    """
    Merges two arrays element-wise using logical OR operation.

    Parameters:
    - arr1 (list): The first input array.
    - arr2 (list): The second input array.

    Returns:
    - list: The merged array where each element is the result of logical OR 
            operation between corresponding elements of arr1 and arr2.

    """
    result = []
    for i in range(min(len(arr1), len(arr2))):
        if arr1[i] or arr2[i]:
            result.append(True)
        else:
            result.append(False)
    return result

def count_true_values(array, count=0):
    count += sum(1 for value in array if value)
    return count

def convert_to_work_status(array_tespit,array_work):
    array_result = [None] * len(array_tespit)
    
    for i in range(len(array_tespit)):
        if array_work[i].any():
            array_result[i] = 1
        elif not array_work[i].any() and array_tespit[i].any():
            array_result[i] = 0
        else:
            array_result[i] = -1
            
    return array_result

def merge_and_convert_to_work_status(cam_108_tespit,cam_108_work,cam_106_tespit,cam_106_work):
    min_len = min(len(cam_108_tespit),len(cam_106_tespit))
    array_result = [None] * min_len
    
    for i in range(min_len):
        if cam_108_work[i].any() and cam_106_tespit[i].any():
            array_result[i] = 2
        elif cam_108_work[i].any() and not cam_106_tespit[i].any():
            array_result[i] = 1
        elif not cam_108_work[i].any() and cam_108_tespit[i].any():
            array_result[i] = 0
        elif not cam_108_tespit[i].any() or (not cam_108_tespit[i].any() and not cam_106_tespit[i].any()):
            array_result[i] = -1
        else:
            array_result[i] = -1   
            
    return array_result

def turn_consecutive_false(arr, threshold):
    consecutive_count = 0
    for i in range(len(arr)):
        if arr[i]:
            consecutive_count += 1
        else:
            if consecutive_count < threshold:
                for j in range(i - consecutive_count, i):
                    arr[j] = False
            consecutive_count = 0
    # Check for consecutive True values at the end of the array
    if consecutive_count < threshold:
        for j in range(len(arr) - consecutive_count, len(arr)):
            arr[j] = False
    return arr

def optimize(input_csv_paths, output_path):
    
    """
    Optimizes the data from input CSV files and writes the processed data to an output CSV file.

    Parameters:
    - input_csv_paths (dict): A dictionary where keys represent the names of datasets ('Makine', 'Insan', etc.) 
                              and values represent the paths to the corresponding CSV files.
    - output_path (str): The path where the optimized data will be saved as a CSV file.

    The function reads data from input CSV files specified in 'input_csv_paths'. It then processes the data 
    to optimize machine and human movement information. Machine movement data is smoothed using a specific 
    algorithm, while human movement data is processed to count and merge certain types of activities. 
    The optimized data is then written to the specified output CSV file, including columns for machine 
    activity, total human work duration, specific types of human activities, and the number of detected 
    workers.

    Note: The function assumes that the input CSV files contain specific columns named 'Makine' for machine 
    movement data, 'Calisan_0' and 'Calisan_1' for two types of human activities, and 'Timestamp' for 
    timestamps of the events.

    Example:
    >>> input_csv_paths = {'Makine': 'path_to_machine_data.csv', 'Insan': 'path_to_human_data.csv'}
    >>> output_path = 'optimized_data.csv'
    >>> optimize(input_csv_paths, output_path)
    """

    # Read data from input CSV files
    csvs_array = {}
    for key, path in input_csv_paths.items():
        csvs_array[key] = pd.read_csv(path, encoding='ISO-8859-9')

    # Extract necessary columns from the CSV data
    makine_veri = csvs_array['Makine']
    insan_tasima_veri = csvs_array['Insan_Tasima']
    insan_kirma_veri = csvs_array['Insan_Kirma']

    makine_hareket_veri = makine_veri['Makine']
    makine_hareket_veri_modified = smoothe(makine_hareket_veri, 10, 3, 10)
    makine_hareket_veri_modified = smoothe(makine_hareket_veri, 100, 1, 10)
    makine_hareket_veri_modified = turn_consecutive_false(makine_hareket_veri_modified,100)
     
    insan_tasima = insan_tasima_veri['Calisiyor']
    insan_kirma = insan_kirma_veri['Calisiyor']
    
    
    insan_sayisi_106 = insan_tasima_veri['Tespit Edilen Insan Sayisi']
    insan_sayisi_108 = insan_kirma_veri['Tespit Edilen Insan Sayisi']

    insan_tespit_108 = insan_sayisi_108 > 0
    insan_tespit_106 = insan_sayisi_106 > 0
    
    insan_tasima_smoothened = smoothe(insan_tasima, 10, 4, 10)
    insan_kirma_smoothened = smoothe(insan_kirma, 10, 4, 10)
    insan_tespit_106 = smoothe(insan_tespit_106, 10, 4, 10)
    insan_tespit_108 = smoothe(insan_tespit_108, 10, 4, 10)
    
    insan_toplam_calisma = merge_work_times(insan_tasima_smoothened, insan_kirma_smoothened)

    tespit_edilen_insan = insan_sayisi_106.add(insan_sayisi_108, fill_value=0)

    durum_108 = convert_to_work_status(insan_tespit_108,insan_kirma_smoothened)
    durum_106 = convert_to_work_status(insan_tespit_106,insan_tespit_106)
    insan_durum = merge_and_convert_to_work_status(insan_tespit_108,insan_kirma_smoothened,insan_tespit_106,insan_tasima_smoothened)        
    
    # Write optimized data to output CSV file
    with open(output_path, 'w', newline='') as csv_file:
        csv_write = csv.writer(csv_file)
        csv_write.writerow(['Timestamp',
                            'Makine_Calisma', 
                            'Insan_Toplam_Calisma',
                            'Tespit_Edilen_Insan_Sayisi', 
                            'Insan_Kirma', 
                            'Insan_Tasima',
                            'Insan_Kirma_Tespit',
                            'Insan_Tasima_Tespit',
                            '108_Durumu',
                            '106_Durumu',
                            'Insan_Durumu'
                            ])

        timestamp_lengths = [len(csvs_array[key]['Timestamp']) for key in csvs_array]
        iteration = min(timestamp_lengths)

        for i in range(iteration):
            csv_write.writerow([i,
                                makine_hareket_veri_modified[i], 
                                insan_toplam_calisma[i],
                                tespit_edilen_insan[i],
                                insan_kirma_smoothened[i], 
                                insan_tasima_smoothened[i],
                                insan_tespit_108[i],
                                insan_tespit_106[i],
                                durum_108[i],
                                durum_106[i],
                                insan_durum[i]
                                ])

## Give Path to CSV's
input_csv_paths = {
    "Makine"    :   "CSV/Machine.csv"  ,
    "Insan_Tasima"     :   "CSV/106.csv"  ,
    "Insan_Kirma" :   "CSV/108.csv"  
}

output_path = "CSV/Merged.csv"


optimize(input_csv_paths, output_path)