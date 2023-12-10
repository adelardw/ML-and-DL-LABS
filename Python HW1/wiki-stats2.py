#! /python3

import argparse
import re
import random

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p','--pause',default = 3)

	args = parser.parse_args()
	res = args.pause
	if res[0] == "-":
		return 'time must be positive'


	n = len(res)
	#standartpause
	if res.isdigit():
		return res

	if 'gauss:' in res:
		mu_with_pref = re.findall(r'\d+\w+', res)[0]
		mu = int(re.findall(r'\d+', mu_with_pref)[0])
		pref = re.sub(r'\d+','', mu_with_pref)
		if pref == 'ms':
			mu *= 10**(-3)

		sigma = float(re.findall(r'\d+(?:\.\d+)?', res)[1])
		gaussian_pause = random.gauss(mu,sigma)
		return gaussian_pause
	#fixedpause
	start, end = re.findall(r'\d+',res)
	start = int(start)
	end = int(end)
	prefixes = re.findall(r'[a-zA-Z]+', res)


	if len(prefixes) == 2 and (prefixes[0] != prefixes[1]):
		if prefixes[0]=='ms':
			start *= 10**(-3)
		else:
			end  *= 10**(-3)

		if start > end:
			return "start must be < end"
		else:
			return 	end - start

	if len(prefixes) == 2 and (prefixes[0] == prefixes[1]):
		if start > end:
			return "start must be < end"
		else:
			return end - start

	if len(prefixes) == 1:
		pos_prefix = res.find(prefixes[0])
		if  (pos_prefix == n - 1 or pos_prefix == n - 2) and (prefixes[0] =='ms'):
			end *= 10**(-3)

		elif prefixes[0] == 's':
			pass
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

			return end - start


			

if  __name__ == "__main__":
	print(main())

