"""
Implementation of the paper 'How to Project Customer Retention', by Peter S. Fader and Bruce G. S. Hardie, from May 2006.
All references to equations refer to the paper here http://www.brucehardie.com/papers/021/sbg_2006-05-30.pdf
"""

import numpy as np
from scipy.optimize import minimize

METHOD = 'Nelder-Mead'


def generate_probabilities(alpha, beta, length):
    """
    Generate probabilities for a customer who fails to renew his contract at the end of the t'th period.
    Implementing the recursive formula in equation (7).
    """
    assert alpha > 0 and beta > 0, "alpha and beta should be grater than 0"
    t = np.arange(1, length)
    t = (beta + t - 1) / (alpha + beta + t)
    t = np.insert(t, 0, alpha / (alpha + beta))
    return np.cumprod(t)


def generate_survivor(probabilities, length):
    """
    Generate the survivor function S at period t=length, the probability for a customer to survives beyond period t.
    Implementing the formula in equation (6), without using the Beta function, instead, it exploits
    the implementation of equation (7).
    """
    return 1 - np.sum(probabilities[:length+1])


def survivor_rates(data):
    """
    The proportion of customers dropping out each year, as required for the log-likelihood function.
    """
    return np.delete(np.insert(data, 0, 1, axis=0), -1) - data


def log_likelihood(alpha, beta, data):
    """
    Logarithm of the likelihood function to be maximized. Obtaining alpha and beta representing the parameters
    for the beta distribution such that the cohort distributed according to the B(alpha, beta) distribution.
    Implementing equation (B3) from the appendix B.
    """
    if alpha <= 0 or beta <= 0:
        return -1000
    survivors = survivor_rates(data)
    probabilities = generate_probabilities(alpha, beta, len(data))
    final_survivor_likelihood = generate_survivor(probabilities, len(data)-1)
    return np.sum(np.multiply(survivors, np.log(probabilities))) + data[-1] * np.log(final_survivor_likelihood)


def log_likelihood_multi_cohort(alpha, beta, data):
    """
    Logarithm of the likelihood function to be maximized. Obtaining alpha and beta representing the parameters
    for the beta distribution such that the cohorts distributed according to the B(alpha, beta) distribution.
    Implementing equation (B3) from the appendix B for multiple (contiguous) cohorts.
    'data' must be a list of cohorts, each with an absolute number per observed time unit.
    """
    total = 0
    for cohort in data:
        total += log_likelihood(alpha, beta, cohort)
    return total


def maximize_multi_cohort(data):
    """
    Maximize the logarithm of the likelihood function for data of multiple cohorts.
    """
    func = lambda params: -log_likelihood_multi_cohort(params[0], params[1], data)
    x0 = np.array([1., 1.])
    res = minimize(func, x0, method=METHOD, options={'xatol': 1e-8})
    return res


def predicted_retention(alpha, beta, t):
    """
    Generate the retention probability r at period t, the probability of customer to be active at the
    end of period t−1 who are still active at the end of period t.
    Implementing the formula in equation (8).
    """
    assert t > 0, "period t should be positive"
    return (beta + t) / (alpha + beta + t)


def predicted_survival(alpha, beta, length):
    """
    Generate prediction of the survival probabilities, the probability for a customer to survives beyond period t.
    Implementing the formula in equation (1).
    """
    assert length > 0, "length should be a positive number"
    if not isinstance(alpha, np.ndarray):
        alpha = np.array(alpha, ndmin=1)
    if not isinstance(beta, np.ndarray):
        beta = np.array(beta, ndmin=1)

    alpha = alpha.reshape(alpha.shape[0], 1)
    beta = beta.reshape(beta.shape[0], 1)

    t = np.arange(length)
    t = (beta + t) / (alpha + beta + t)
    return np.cumprod(t, axis=1)


def fit(data):
    """
    Fitting parameters alpha and beta of the Beta distribution to the given data such that the logarithm of
    the likelihood function is maximized.
    """
    return fit_multi_cohort([data])


def fit_multi_cohort(data):
    """
    Fitting parameters alpha and beta of the Beta distribution to the given multi-cohort data such that
    the logarithm of the likelihood function is maximized.
    """
    res = maximize_multi_cohort(data)
    return res.x if res.status == 0 else None


def test(data, length):
    data_trunc = data[:length]
    alpha, beta = fit(data_trunc)
    pred = predicted_survival(alpha, beta, data.shape[0])
    print(f'alpha = {alpha}, beta = {beta}')
    with np.printoptions(precision=1, suppress=True):
        print(f'real: {data * 100}')
        print(f'pred: {pred * 100}')


def test_regular_paper_data():
    """Test against the regular subscription retention data from the paper"""
    real = np.array([.631, .468, .382, .326, .289, .262, .241, .223, .207, .194, .183, .183])
    test(real, 7)


def test_high_end_paper_data():
    """Test against the High End subscription retention data from the paper"""
    real = np.array([.869, .743, .653, .593, .551, .517, .491, .468, .445, .427, .409, .394])
    test(real, 7)


def test_random_cohort(alpha=3.14159, beta=2.71828, n_days=14):
    example_data = predicted_survival(alpha, beta, n_days - 1)[0]
    alpha_p, beta_p = fit(example_data)
    print(f'alpha = {alpha}, and found alpha = {alpha_p}')
    print(f'beta = {beta}, and found beta = {beta_p}')


def test_random_multi_cohort_data(alpha=3.14159, beta=2.71828, n_days=14, n_cohorts=5):
    alpha_array = alpha + np.random.normal(0, 0.01, n_cohorts)
    beta_array = beta + np.random.normal(0, 0.01, n_cohorts)

    example_data = predicted_survival(alpha_array, beta_array, n_days - 1)

    alpha_p, beta_p = fit_multi_cohort(example_data)
    print(f'alpha = {alpha}, and found alpha = {alpha_p}')
    print(f'beta = {beta}, and found beta = {beta_p}')


def main():
    test_regular_paper_data()
    test_high_end_paper_data()
    test_random_cohort()
    test_random_multi_cohort_data()


if __name__ == '__main__':
    main()
