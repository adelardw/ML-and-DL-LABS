#! /bin/python3

import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('page')
	args = parser.parse_args()
	return args.page

if __name__ == "__main__":
	print(main())

