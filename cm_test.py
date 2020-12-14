import os
import sys

def main(data_path: str):
    configurations = {
        "one": ["4", "1", "512", "0.1", "0.5"],
        "two": ["3", "1", "512", "0.1", "0.5"],
        "three": ["6", "1", "512", "0.1", "0.5"],
        "four": ["4", "0", "512", "0.1", "0.5"],
        "five": ["4", "2", "512", "0.1", "0.5"],
        "six": ["4", "1", "256", "0.1", "0.5"],
        "seven": ["4", "1", "1024", "0.1", "0.5"],
        "eight": ["4", "1", "512", "0.05", "0.5"],
        "nine": ["4", "1", "512", "0.1", "0.7"]
    }

    exec_train = "./predictor train {window_size} {path} {fit} {population} {mut} {cross} {output}"
    exec_test = "./predictor test {window_size} {path} {fit} {weights} {output}"

    for c in range(1,2):
        for k,v in configurations.items():
            os.system(
                exec_train.format(
                    window_size=v[0],
                    path=data_path + "Train/" + str(c),
                    fit=v[1],
                    population=v[2],
                    mut=v[3],
                    cross=v[4],
                    output=f"{c}_{k}"
                )
            )

    for c in range(1,2):
        for k,v in configurations.items():
            os.system(
                exec_test.format(
                    window_size=v[0],
                    path=data_path + "Test/" + str(c),
                    fit=v[1],
                    weights=f"{c}_{k}.weights",
                    output=f"{c}_{k}"
                )
            )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Arguments missing")
    
    main(sys.argv[1])
