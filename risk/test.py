# %%
import pandas as pd
import pickle as pkl
import os
BASE_DIR = os.path.dirname(__file__)

# Build an absolute path to the pickle file
mop_path = os.path.join(BASE_DIR, 'model', 'mop_pl.pkl')
fop_path = os.path.join(BASE_DIR, 'model', 'fop_pl.pkl')
mhos_path = os.path.join(BASE_DIR, 'model', 'mhos_pl.pkl')
fhos_path = os.path.join(BASE_DIR, 'model', 'fhos_pl.pkl')

mop = pkl.load(open(mop_path,'rb'))
fop = pkl.load(open(fop_path, 'rb'))
mhos = pkl.load(open(mhos_path, 'rb'))
fhos = pkl.load(open(fhos_path, 'rb'))
# %%
def predict(df):
    for i in ['sex', 'intubated', 'pneumonia', 'pregnancy',
       'diabetes', 'copd', 'asthma', 'inmsupr', 'hypertension',
       'other_disease', 'cardiovascular', 'obesity', 'renal_chronic',
       'tobacco', 'contact_other_covid', 'icu']:
       df[i] = pd.Categorical(df[i].astype(int))
    df['age'] = df['age'].astype(int)
    if (df['sex'][0]==1):
        df.drop(columns = ['sex'], inplace=True)
        if (df['icu'][0]==1 or df['intubated'][0]==1):
            p = fhos.predict(df)
            print('fhos')
        else:
            df.drop(columns=['icu', 'intubated'], inplace=True)
            p = fop.predict(df)
            print('fop')
    else:
        df.drop(columns=['pregnancy','sex'], inplace=True)
        print(df['intubated'][0]==1)
        print(df['intubated'][0])
        if (df['icu'][0]==1 or df['intubated'][0]==1):
            p = mhos.predict(df)
            print('mhos')
        else:
            df.drop(columns=['icu', 'intubated'], inplace=True)
            p = mop.predict(df)
            print('mop')
    print(' model predicted :'+str(p[0]))
    return p[0]


import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1) Define fuzzy variables
severity = ctrl.Antecedent(np.arange(0, 101, 1), 'severity')
risk     = ctrl.Antecedent(np.arange(1, 3, 1), 'risk')
overall  = ctrl.Consequent(np.arange(0, 101, 1), 'overall')

# 2) Membership functions for severity
severity['low']    = fuzz.trimf(severity.universe, [0, 0, 50])
severity['medium'] = fuzz.trimf(severity.universe, [25, 50, 75])
severity['high']   = fuzz.trimf(severity.universe, [50, 100, 100])

# 3) Membership functions for risk
risk['high_risk']  = fuzz.trimf(risk.universe, [1, 1, 2])
risk['lower_risk'] = fuzz.trimf(risk.universe, [1, 2, 2])

# 4) Membership functions for overall
overall['low']      = fuzz.trimf(overall.universe, [0, 0, 25])
overall['moderate'] = fuzz.trimf(overall.universe, [20, 40, 60])
overall['high']     = fuzz.trimf(overall.universe, [50, 70, 90])
overall['critical'] = fuzz.trimf(overall.universe, [80, 100, 100])

# 5) Define fuzzy rules—note how each consequent is overall['label']
rules = [
    ctrl.Rule(severity['low'] & risk['lower_risk'], overall['low']),
    ctrl.Rule(severity['low'] & risk['high_risk'],  overall['moderate']),
    ctrl.Rule(severity['medium'] & risk['lower_risk'], overall['moderate']),
    ctrl.Rule(severity['medium'] & risk['high_risk'],  overall['high']),
    ctrl.Rule(severity['high'] & risk['lower_risk'], overall['high']),
    ctrl.Rule(severity['high'] & risk['high_risk'],  overall['critical']),
]


severity_ctrl = ctrl.ControlSystem(rules)
severity_sim  = ctrl.ControlSystemSimulation(severity_ctrl)

def compute_overall(sev_score, risk_code):
    print("inputs to fuzzy",'\t',sev_score, '\t', risk_code)
    severity_sim.input['severity'] = sev_score
    severity_sim.input['risk']     = risk_code
    severity_sim.compute()
    crisp = severity_sim.output['overall']

    memberships = {
        label: fuzz.interp_membership(overall.universe,
                                     overall[label].mf,
                                     crisp)
        for label in overall.terms
    }
    label = max(memberships, key=memberships.get).capitalize()
    return crisp, label


'''for sev, rk in [(20,2), (70,1), (60,1), (90,1)]:
    crisp, lab = compute_overall(sev, rk)
    print(f"Severity={sev}%  Risk={rk}  →  Overall={crisp:.1f}, Label={lab}")'''
