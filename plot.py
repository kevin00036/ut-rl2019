import numpy as np
import json
import matplotlib.pyplot as plt
import sys




def main():
    path = sys.argv[1]

    print(path)

    obj = json.load(open(path))


    log_steps = obj['info']['aggregate_steps']
    total_steps = obj['info']['steps']

    varlist = ['IntR', 'IntR_Est']
    # varlist = ['DiscExtR', 'DiscExtR_Est']
    varlist = ['Plan_ExtR', 'Plan_DiscExtR', 'Plan_DiscExtR_Est']

    for var in varlist:
        varobj = obj['data'][var]
        x = [v[0] for v in varobj]
        y = [v[1]['mean'] for v in varobj]

        plt.plot(x, y)
    plt.show()




if __name__ == '__main__':
    main()
