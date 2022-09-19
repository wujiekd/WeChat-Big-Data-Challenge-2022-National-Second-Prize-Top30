import pandas as pd
from config import parse_args
def hebin():
    args = parse_args()
    csv1 = pd.read_csv(args.test_output_csv2+"0.csv",header=None,dtype=str)
    csv2 = pd.read_csv(args.test_output_csv2+"1.csv",header=None,dtype=str)
    all_data = pd.concat([csv1,csv2])
    
    all_data.to_csv(args.test_output_csv,index=None,header=None)
    print(all_data.shape)
    

if __name__ == '__main__':
    hebin()
