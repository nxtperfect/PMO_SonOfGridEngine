import random
import multiprocessing as mp


def f(x, y):
    return x * x + y * y


def worker(samples):
    partial_sum = 0.0

    for _ in range(samples):
        x = random.random()
        y = random.random()
        partial_sum += f(x, y)

    return partial_sum


if __name__ == "__main__":

    total_samples = 1_000_000
    processes = 4

    samples_per_process = total_samples // processes

    with mp.Pool(processes) as pool:
        results = pool.map(worker, [samples_per_process] * processes)

    total = sum(results)

    estimate = total / total_samples

    print("Estimated integral:", estimate)
