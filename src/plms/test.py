import os
root_dir = "/home/qnnguyen/stance-detection/code/src/plms/results/pstance/frameaxis/biden"
file_set = set()
df = []
for dir_, _, files in os.walk(root_dir):
#     for file_name in files:
#         rel_dir = os.path.relpath(dir_, root_dir)
#         rel_file = os.path.join(rel_dir, file_name)
#         file_set.add(rel_file)
# print(file_set)
    df.append(dir_)
print(df)
df.sort()
print(d)
# print(df[1])