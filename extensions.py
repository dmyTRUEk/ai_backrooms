# Extension functions

from datetime import timedelta
from random import random



def string_multisplit(string: str, seps: list[str]) -> list[str]:
	prev = [string]
	for sep in seps:
		next = []
		for s in prev:
			next.extend(s.split(sep))
		prev = next
	return [s for s in prev if s != '']



def time_to_my_format(time: timedelta) -> str:
	d = time.days
	h = (time.seconds // (60*60)) % 24
	m = (time.seconds // 60) % 60
	s = (time.seconds % 60)
	ms = time.microseconds // 1000
	if d > 0:
		return f'{d}d {h}h {m}m'
	elif h > 0:
		return f'{h}h {m}m {s}s'
	elif m > 0:
		return f'{m}m {s}s'
	else:
		return f'{s}s {ms}ms'



def to_base(n: int, *, digits: str) -> str:
	BASE = len(digits)
	if n == 0: return '0'
	is_neg = n < 0
	n = abs(n)
	s = ''
	while n != 0:
		digit = n % BASE
		digit_s = digits[digit]
		s = digit_s + s
		n //= BASE
	if is_neg:
		s = '-' + s
	return s

def to_base36(n: int) -> str:
	return to_base(n, digits='0123456789abcdefghijklmnopqrstuvwxyz')



def shuffled(l: list) -> list:
	return sorted(l, key=lambda _: random())

