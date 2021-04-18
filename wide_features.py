from helper_code import *
# from model_funcs import *
# from data_funcs import *

import sys

if __name__ == '__main__':

    data_directory = sys.argv[1]

    header_files, _ = find_challenge_files(data_directory)

    age_count = 0
    sex_count = 0
    total_count = 0
    age_nan = 0
    sex_nan = 0

    for file in header_files:
        header = load_header(file)

        total_count += 1

        age = get_age(header)
        sex = get_sex(header)

        if age:
            age_count += 1
            print(age)
        else:
            age_nan += 1


        if sex:
            sex_count += 1
        else:
            sex_nan +=1 

    print('Ages', age_count, age_nan)
    print('Sexes', sex_count, sex_nan)
    print('Total', total_count)