import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.normal(0,1,n)
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequencies")
    plt.title("Normal Histogram")
    plt.show()
    return data

def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.uniform(0,10,n)
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequencies")
    plt.title("Uniform distribution")
    plt.show()
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot a histogram with 10 bins (with labels + title),
    and return the generated data.
    """
    data = np.random.binomial(1, 0.5, n)
    plt.hist(data, bins=10)
    plt.xlabel("Values")
    plt.ylabel("Frequencies")
    plt.title("Bernoulli distribution")
    plt.show()
    return data


# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    return np.sum(data) / len(data)


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    n = len(data)
    mean = sample_mean(data)
    return np.sum((data - mean)**2) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Use a consistent quartile definition. The tests for the fixed
    dataset [5,1,3,2,4] expect Q1=2 and Q3=4.
    """
    arr = list(data)
    n = len(arr)
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp

    minimum = arr[0]
    maximum = arr[-1]

    #median
    if n % 2 == 1:
        median = arr[n // 2]
    else:
        median = (arr[n // 2 - 1] + arr[n // 2]) / 2

    #quartiles
    q1_index = n // 4
    q3_index = (3 * n) // 4

    q1 = arr[q1_index]
    q3 = arr[q3_index]

    return minimum, maximum, median, q1, q3


# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.array(x)
    y = np.array(y)
    
    n = len(x)
    mx = sample_mean(x)
    my = sample_mean(y)
    
    return np.sum((x - mx)*(y - my)) / (n - 1)


# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    cov_xy = sample_covariance(x, y)
    var_x = sample_variance(np.array(x))
    var_y = sample_variance(np.array(y))
    
    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
