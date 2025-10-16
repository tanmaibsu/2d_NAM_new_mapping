import csv

with open("double_bit_test_24_12_5_ior.csv", newline='') as f:
    reader = list(csv.reader(f))
    header = reader[0]
    columns = list(zip(*reader[1:]))

    for i, col in enumerate(columns):
        if all(value == col[0] for value in col):
            print(f"✅ All values in column '{header[i]}' are the same.")
        else:
            print(f"❌ Column '{header[i]}' has different values.")

