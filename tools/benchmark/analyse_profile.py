import argparse, json
from prettytable import PrettyTable

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', default='profile.json', type=str)
    parser.add_argument('--output', '-o', default='profile_out.txt', type=str)
    args = parser.parse_args()

    json_path = args.json
    with open(json_path) as f:
        profile_data = json.load(f)
    layer_data = []
    for data in profile_data:
        if type(data) is dict and 'name' in data:
            layer_data.append([data['name'], float(data['averageMs']), float(data['percentage'])])
    
    table = PrettyTable(['layer', 'time(averageMs)', 'percentage'])
    table.title = 'TensorRT Analyse'
    layer_data = sorted(layer_data, key=lambda x:float(x[1]), reverse=True)
    table.add_rows(layer_data)

    with open(args.output, 'w+') as f:
        f.write(str(table))
    
    print(f'result save in {args.output}')