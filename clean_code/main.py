from histogram import Histogram
from utils import *
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)

def main(filename=None):
    if filename is not None:
        input_reader = Input(filename)
    else:
        input_reader = Input()
    print(input_reader)
    
    results = compute_resonances_total(input_reader, input_reader.length, input_reader.occupation_probability)
    histogram = Histogram(results, input_reader)
    histogram.plot()
    return results
if __name__ == '__main__':
    filename = "clean_code/inputs.json"
    input_reader = Input(filename)
    results = compute_resonances_total(input_reader, input_reader.length, input_reader.occupation_probability)
    import csv
    with open('clean_code/results.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow([f"length: {input_reader.length}, p: {input_reader.occupation_probability}", ""])
        writer.writerow(["energy", "width", "a_eff"])
        for result in results:
            energy  = result.energy
            a_eff = result.a_eff
            width = result.width
            writer.writerow([energy, width, a_eff])