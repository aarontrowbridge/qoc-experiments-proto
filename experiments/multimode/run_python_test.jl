using PyCall

@pyinclude "experiments/multimode/run_python_test.py"

us = [randn(3) for t = 1:3]

ys = py"test"(us)

display(ys)

sum.(us) ≈ ys
