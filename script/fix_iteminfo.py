import sys
args = len(sys.argv)
print('args #{}'.format(args))
inpf =  sys.argv[1] if args>=2 else 'data/item-info-raw'
outpf = sys.argv[2] if args>=3 else 'data/item-info'
input = open(inpf, 'r')
output = open(outpf, 'w')

item_map = {}
unknown_count = 0
for line in input.readlines():
    str = line.strip()
    arr = str.split('\t')
    if arr[0] not in item_map or "UNKNOWN" in item_map[arr[0]] or not item_map[arr[0]]:
        item_map[arr[0]] = arr[1]
    else:
        unknown_count += 1

print("UNKNOWN category count # {}".format(unknown_count))

for key in item_map:
    output.write("{}\t{}\n".format(key, item_map[key]))
output.close()
print('ALL DONE! # fix_iteminfo.py')


# item_map = {}
# error_count = 0
# for line in input.readlines():
#     str = line.strip()
#     arr = str.split('\t')
#     if arr[0] not in item_map:
#         item_map[arr[0]] = arr[1]
#     else:
#         print("Error, id duped, id: {}, cate: {}".format(arr[0], arr[1]))
#         error_count += 1
#
# print("error_count count # {}".format(error_count))
# print("ALL DONE!")


# for line in input.readlines():
#     str = line.strip()
#     arr = str.split('\t')
#     if len(arr)==2 and arr[1]:
#         output.write(line)
#     elif len(arr)==2 and arr[1] or len(arr)==1:
#         output.write('{}\t{}\n'.format(arr[0],"NEWS_UNKNOWN"))
#
# output.close()
# print('ALL DONE!')