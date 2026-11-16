"""Run only when ready; creates actual experimental outputs, never prefilled results."""
from pathlib import Path
import sys,json,time,argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import SugenoANFIS
ROOT=Path(__file__).resolve().parents[1]

def run(seed=42):
    df=pd.read_csv(ROOT/'data/table2_transcribed.csv')
    X=df[['moisture','volatile','fixed_carbon','ash']].to_numpy();y=df.calorific_kj_g.to_numpy()
    ix=np.arange(len(df)); tr,te=train_test_split(ix,test_size=.2,random_state=seed)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[]; preds=pd.DataFrame({'row_index':te,'observed':y[te],'origin':df.origin.iloc[te].to_numpy()})
    for name,k in [('baseline_81_rules',3),('proposed_16_rules',2)]:
        model=SugenoANFIS(mfs_per_input=k,ridge=.01,max_iter=30,seed=seed)
        t=time.perf_counter();model.fit(X[tr],y[tr]);seconds=time.perf_counter()-t
        p=model.predict(X[te]);preds[name]=p
        rows.append({'model':name,'rules':model.rule_count,'n_train':len(tr),'n_test':len(te),'rmse_kj_g':float(np.sqrt(mean_squared_error(y[te],p))),'mae_kj_g':float(mean_absolute_error(y[te],p)),'r2':float(r2_score(y[te],p)),'fit_seconds':seconds,'seed':seed})
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);preds.to_csv(out/'test_predictions.csv',index=False)
    plt.figure(figsize=(6,5))
    for name in ['baseline_81_rules','proposed_16_rules']:
        plt.scatter(preds.observed,preds[name],label=name)
    lo=min(preds.observed.min(),preds.iloc[:,3:].min().min());hi=max(preds.observed.max(),preds.iloc[:,3:].max().max())
    plt.plot([lo,hi],[lo,hi],'k--');plt.xlabel('Measured (kJ/g)');plt.ylabel('Predicted (kJ/g)');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(out/'prediction_comparison.png',dpi=180);plt.close()
    (out/'run_metadata.json').write_text(json.dumps({'seed':seed,'dataset':'paper Table 2 manually transcribed; verify against PDF','split':'random 80/20','input_order':['moisture','volatile','fixed_carbon','ash'],'note':'Not a full reproduction of MATLAB training or published numeric results'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=42);args=parser.parse_args();run(args.seed)
