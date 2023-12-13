import pandas as pd
import numpy as np
import openai
import random
import time


def predict(x: pd.DataFrame, llm_fn, verbose: bool = True, mojito: bool = False):
    count = 0
    xcs = []
    for idx in range(len(x)):
        xc = x.iloc[[idx]].copy()
        ltuple, rtuple = get_tuples(xc)
        answer = llm_fn.er(ltuple, rtuple)
        if verbose:
            print(f'{ltuple}\n{rtuple}')
            print(answer)
        nomatch_score, match_score = text_to_match(answer, llm_fn)
        xc['nomatch_score'] = nomatch_score
        xc['match_score'] = match_score
        count += 1
        if mojito:
            full_df = np.dstack((xc['nomatch_score'], xc['match_score'])).squeeze()
            xc = full_df
        xcs.append(xc)
        # print(f'{count},{self.summarized},{self.idks}')
    return pd.concat(xcs, axis=0)


def get_tuples(xc):
    elt = dict()
    ert = dict()
    for c in xc.columns:
        if c in ['ltable_id', 'rtable_id']:
            continue
        if c.startswith('ltable_'):
            elt[str(c)] = xc[c].astype(str).values[0]
        if c.startswith('rtable_'):
            ert[str(c)] = xc[c].astype(str).values[0]
    return elt, ert


def text_to_match(answer, llm_fn, n=0):
    summarized = 0
    idks = 0
    no_match_score = 0
    match_score = 0
    if answer.lower().startswith("yes"):
        match_score = 1
    elif answer.lower().startswith("no"):
        no_match_score = 1
    else:
        if "yes".casefold() in answer.casefold():
            match_score = 1
        elif "no".casefold() in answer.casefold():
            no_match_score = 1
        elif n == 0:
            template = "summarize \"response\" as yes or no"
            try:
                summarized_answer = llm_fn(template.replace("response", answer))
            except:
                summarized_answer = "false"
            summarized += 1
            snms, sms = text_to_match(summarized_answer, llm_fn, n=1)
            if snms == 0 and sms == 0:
                idks += 1
                no_match_score = 1
    return no_match_score, match_score


def read_prompt(file_path: str):
    with open(file_path) as file:
        lines = [(line.rstrip().split(':')) for line in file]
    return lines


def concordance_correlation(y_pred, y_true):
    # Raw data
    dct = {
        'y_true': y_true,
        'y_pred': y_pred
    }
    df = pd.DataFrame(dct)
    # Remove NaNs
    df = df.dropna()
    # Pearson product-moment correlation coefficients
    y_true = df['y_true']
    y_pred = df['y_pred']
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def completion_with_backoff(deployment_id="gpt-35-turbo", model="gpt-3.5-turbo", messages=None, temperature=0,
                            initial_delay=1, max_retries=10, exponential_base: float = 2, jitter: bool = True,
                            errors: tuple = (openai.RateLimitError, openai.Timeout), ):
    num_retries = 0
    delay = initial_delay

    while True:
        try:
            openai.api_type = "azure"
            openai.api_version = "2023-05-15"
            return openai.ChatCompletion.create(deployment_id=deployment_id, model=model, messages=messages,
                                                temperature=temperature)

        # Retry on specified errors
        except errors as e:
            print(e)
            # Increment retries
            num_retries += 1

            # Check if max retries has been reached
            if num_retries > max_retries:
                raise Exception(
                    f"Maximum number of retries ({max_retries}) exceeded."
                )

            # Increment the delay
            delay *= exponential_base * (1 + jitter * random.random())

            # Sleep for the delay
            time.sleep(delay)

        # Raise exceptions for any errors not specified
        except Exception as e:
            raise e
