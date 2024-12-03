import os
import shutil
import pandas as pd
import cv2

def process_data(excel_file_path, dataset_parent_dir, output_folder, dataset_name, part=None):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(excel_file_path)
    
    # Initialize the new dataframe based on the dataset name
    if dataset_name.lower() == "casme2":
        newdataframe = df.loc[:, ['OnsetFrame', 'ApexFrame', 'Filename', 'Subject', 'Estimated Emotion', 'OffsetFrame']]
        dataset_parent_dir = os.path.join(dataset_parent_dir, "CASME2", "CASME2-RAW", "sub")
    elif dataset_name.lower() == "casme^2":
        newdataframe = df.loc[:, ['Onset', 'Apex', 'Folders', 'Subject', 'Emotion', 'Offset', 'Emotion Type', 'Category']]
        dataset_parent_dir = os.path.join(dataset_parent_dir, "selectedpic", "selectedpic", "s")
    elif dataset_name.lower() == "casme^3":
        if part == 'A':
            newdataframe = df.loc[:, ['Onset', 'Apex', 'Filename', 'Subject', 'emotion', 'Offset']]
            dataset_parent_dir = os.path.join(dataset_parent_dir, "part_A", "data", "part_A")
        elif part == 'C':
            newdataframe = df[['onset', 'sub', 'emotion', 'offset']]
            newdataframe.insert(1, 'placeholder1', '')
            newdataframe.insert(2, 'placeholder2', '')
            dataset_parent_dir = os.path.join(dataset_parent_dir, "part_C", "RGB_Depth", "part_C", "part_C")
    elif dataset_name.lower() == "samm":
        newdataframe = df.loc[:, ['Onset Frame', 'Apex Frame', 'Filename', 'Subject', 'Estimated Emotion', 'Offset Frame']]
        dataset_parent_dir = os.path.join(dataset_parent_dir, "SAMM")
    else:
        raise ValueError("Unsupported dataset name")
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Call the folder_extractor function
    folder_extractor(dataset_name, newdataframe, output_folder, dataset_parent_dir, part)

def folder_extractor(name, newdataframe, parent_frame_path, dataset_parent_dir, part=None):
    count = 1
    for data in range(0, newdataframe.shape[0]):
        dataseries = newdataframe.iloc[data]
        if name == "casme2" or name == "casme^2":
            if name == "casme2":
                if dataseries.values[3] < 10:
                    sub = '0' + str(dataseries.values[3])
                else:
                    sub = str(dataseries.values[3])
                folder_path = os.path.join(dataset_parent_dir+sub, dataseries.values[2])   
            elif name == "casme^2":           
                actual_number = int(dataseries.values[3]) + 14
                if dataseries.values[6] == 'micro-expression':
                    sub = str(actual_number)
                else:
                    continue
                folder_path = os.path.join(dataset_parent_dir+sub, dataseries.values[2])

            class_target_folder = os.path.join(parent_frame_path, str(dataseries.values[4]))          
            if not os.path.exists(class_target_folder):
                os.makedirs(class_target_folder)
            new_folder_name = f"{dataseries.values[0]}_{dataseries.values[1]}_{dataseries.values[5]}_{dataseries.values[2]}"
            subject_target_folder = os.path.join(class_target_folder, new_folder_name)
            if os.path.exists(folder_path):
                if not os.path.exists(subject_target_folder):
                    print("Copying folder: " + subject_target_folder)
                    shutil.copytree(folder_path, subject_target_folder)
                else:
                    print("Folder already exists: " + subject_target_folder)
        elif name == "casme^3" or name == "samm":     
            if name == "casme^3":
                if part == 'A':
                    sub = str(dataseries.values[3])
                    folder_path = os.path.join(dataset_parent_dir, sub, dataseries.values[2], "color")
                    new_folder_name = f"{dataseries.values[0]}_{dataseries.values[1]}_{dataseries.values[5]}_{dataseries.values[2]}"
                elif part == 'C':  
                    if dataseries.values[3] < 10:
                        sub = '0' + str(dataseries.values[3])
                    else:
                        sub = str(dataseries.values[3])
                    folder_path = os.path.join(dataset_parent_dir, sub, "a", "color")
                    new_folder_name = f"{dataseries.values[0]}_{dataseries.values[5]}"
            if name == "samm":
                if dataseries.values[3] < 10:
                    sub = '00' + str(dataseries.values[3])
                else:
                    sub = '0' + str(dataseries.values[3])
                new_folder_name = f"{dataseries.values[0]}_{dataseries.values[1]}_{dataseries.values[5]}_{dataseries.values[2]}"
                folder_path = os.path.join(dataset_parent_dir, sub)
            for frame_number in range(dataseries.values[0], dataseries.values[5]+1):  
                if os.path.exists(folder_path):
                    class_target_folder = os.path.join(parent_frame_path, dataseries.values[4])
                    if not os.path.exists(class_target_folder):
                        os.makedirs(class_target_folder)
                    subject_target_folder = os.path.join(class_target_folder, new_folder_name) 
                    if not os.path.exists(subject_target_folder):
                        os.makedirs(subject_target_folder)
                    if name == 'casme^3':
                        frame_path = os.path.join(folder_path, str(frame_number) + ".jpg")
                    elif name == 'samm':
                        frame_path = os.path.join(folder_path, dataseries.values[2], sub + "_0"+ str(frame_number) + ".jpg")
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        frame_name = str(frame_number) + ".png"
                        frame_target_path = os.path.join(subject_target_folder, frame_name)
                        cv2.imwrite(frame_target_path, frame)
                    else:
                        print("Frame " + str(frame_number) + " not found in " + frame_path)
                
            else:
                print("Folder path does not exist: " + folder_path)
        count = count + 1
