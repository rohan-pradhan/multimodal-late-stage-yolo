import os 
import argparse 



def rename_files(thermal_input_path, vison_input_path):
    thermal_file_list = os.listdir(thermal_input_path)
    counter = 0
    for f_t in thermal_file_list:
        f_v = f_t.split(".")[0] + ".jpg" # replace .jpeg file ending to .jpg file ending 
        if os.path.isfile(vison_input_path+"/"+f_v): 
            thermal_file_old = thermal_input_path + "/" + f_t
            thermal_file_new = thermal_input_path + "/" + str(counter) + ".jpeg"

            vison_file_old = vison_input_path + "/" + f_v
            vison_file_new = vison_input_path + "/" + str(counter) + ".jpg"


            os.rename(thermal_file_old, thermal_file_new)
            os.rename(vison_file_old, vison_file_new)
            counter +=1
        
        else: 
            os.remove(thermal_input_path+"/"+f_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thermal_input', type=str, default='test_data/thermal', help='thermal input file path')
    parser.add_argument('--vison_input', type=str, default='test_data/vison', help='vison input file path')
    opt = parser.parse_args()
    print (opt)

    rename_files(opt.thermal_input, opt.vison_input)





