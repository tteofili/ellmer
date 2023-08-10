import pandas as pd
import numpy as np


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
    elt = []
    ert = []
    for c in xc.columns:
        if c in ['ltable_id', 'rtable_id']:
            continue
        if c.startswith('ltable_'):
            elt.append(str(c) + ':' + xc[c].astype(str).values[0])
        if c.startswith('rtable_'):
            ert.append(str(c) + ':' + xc[c].astype(str).values[0])
    ltuple = '\n'.join(elt)
    rtuple = '\n'.join(ert)
    return ltuple, rtuple


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
            template = "summarize response as yes or no"
            summarized_answer = llm_fn(template.replace("response", answer))
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
