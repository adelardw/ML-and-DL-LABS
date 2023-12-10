#! bin/python2

import random
import re
import argparse
from typing import Any,Union

def type_pauses(type_pause: str, param:Any)-> Union[int,float]:

    """
    Manual:

    type_pause: 3 names "fixed","interval","gauss"
    param: "fixed" may be,for example, "100//100ms//100s"; "interval" may be,for example, "100ms-200ms//100-200ms//100ms-200//100ms-200s//100-200s//...and another//"
            "gauss" may be,for example "123/4" in general things -> mu/std dev
    return time in seconds [s] ( type float or int )
    Also: if params dont have prefix,then this params have based prefix -ms
    """
    if type_pause == "fixed":
        num = re.findall(r'\d+',param)[0]
        num = float(num)
        pref = re.sub(r'\d+','',param)
        if pref == 's' and len(pref) !=0:
            return num
        if pref == 'ms' or len(pref) ==0:
            num *= 10**(-3)
            return num

    if type_pause =="interval":
        n = len(param)
        start, end = re.findall(r'\d+',param)
        start = int(start)
        end = int(end)
        prefixes = re.findall(r'[a-zA-Z]+', param)

        if len(prefixes) == 2 and (prefixes[0] != prefixes[1]):

            if prefixes[0] == "ms":
                start *= 10**(-3)
            else:
                end  *= 10**(-3)

            if start > end:
                return "start must be < end"
            else:
                return end - start

        if len(prefixes) == 2 and (prefixes[0] == prefixes[1]):

            if start > end:
                return "start must be < end"
            else:
                return end - start

        if len(prefixes) == 1:

            pos_prefix = param.find(prefixes[0])
            if  prefixes[0] =='ms':
                start *= 10**(-3)
                end *= 10**(-3)

            if prefixes[0] == 's':
                if pos_prefix == n - 1:
                    end *= 10**(-3)
                else:
                    start *= 10**(-3)

            if start > end:

                return "start must be < end"
            else:
                return end - start

        else:
            if start > end:
                return "start must be < end"
            else:
                return (end - start)*0.001

    if type_pause == "gauss":
        mu_with_pref = re.findall(r'\d+\w+', param)

        if len(mu_with_pref) !=0 and re.sub(r'\d+','', mu_with_pref[0]) !='ms':
            mu = float(re.findall(r'\d+', mu_with_pref[0])[0])
            sigma = float(re.findall(r'\d+(?:\.\d+)?', param)[1])
            gaussian_pause = random.gauss(mu,sigma)
            return gaussian_pause

        else:
            mu,sigma = re.findall(r'\d+\.?\d*', param)
            mu = float(mu)
            sigma = float(sigma)
            gaussian_pause = random.gauss(mu,sigma)
            gaussian_pause *= 0.001
            return gaussian_pause

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pause",default = "fixed")
    parser.add_argument("--params",default = 3)
    args = parser.parse_args()
    paused = args.pause
    parameters = args.params
    w = type_pauses(type_pause = paused, param = parameters)
    return w


if __name__ == "__main__":
	print(main())
