"""Transparent first-order Sugeno neuro-fuzzy approximation (not MATLAB ANFIS equivalence)."""
import itertools
import numpy as np
from scipy.optimize import minimize

class SugenoANFIS:
    def __init__(self, mfs_per_input=3, ridge=0.01, max_iter=30, seed=42):
        self.mfs_per_input=mfs_per_input; self.ridge=ridge; self.max_iter=max_iter; self.seed=seed

    def _features(self, X, centers, widths):
        # X is normalized to [0,1] using training-set ranges only.
        mu=np.exp(-0.5*((X[:,:,None]-centers[None,:,:])/widths[None,:,:])**2)
        ids=np.array(list(itertools.product(range(self.mfs_per_input),repeat=X.shape[1])))
        w=np.ones((len(X),len(ids)))
        for d in range(X.shape[1]):w*=mu[:,d,ids[:,d]]
        w=w/(w.sum(axis=1,keepdims=True)+1e-12)
        z=np.column_stack([np.ones(len(X)),X]); return (w[:,:,None]*z[:,None,:]).reshape(len(X),-1)

    def fit(self,X,y):
        X=np.asarray(X,float);y=np.asarray(y,float).reshape(-1)
        self.xmin=X.min(axis=0);self.span=np.maximum(X.max(axis=0)-self.xmin,1e-9)
        xn=np.clip((X-self.xmin)/self.span,0,1)
        self.ymean=y.mean();self.ystd=max(y.std(),1e-9);yn=(y-self.ymean)/self.ystd
        k=self.mfs_per_input
        c=np.tile(np.linspace(0,1,k),(X.shape[1],1)); widths=np.full_like(c, max(0.2,1/(k-1)))
        self.centers=c;self.widths=widths
        # Alternating optimization: ridge consequent LS and Gaussian premise centers.
        def consequent(cent):
            phi=self._features(xn,cent,widths)
            a=phi.T@phi+self.ridge*np.eye(phi.shape[1]); b=phi.T@yn
            return np.linalg.solve(a,b)
        for _ in range(self.max_iter):
            coef=consequent(c)
            def loss(flat):
                cc=flat.reshape(c.shape);pred=self._features(xn,cc,widths)@coef
                return np.mean((pred-yn)**2)+0.002*np.mean((cc-np.tile(np.linspace(0,1,k),(X.shape[1],1)))**2)
            res=minimize(loss,c.ravel(),method='L-BFGS-B',bounds=[(0,1)]*c.size,options={'maxiter':8})
            c=res.x.reshape(c.shape)
        self.centers=c;self.coef=consequent(c);return self

    def predict(self,X):
        X=np.asarray(X,float);X=np.atleast_2d(X)
        xn=np.clip((X-self.xmin)/self.span,0,1)
        return (self._features(xn,self.centers,self.widths)@self.coef)*self.ystd+self.ymean

    @property
    def rule_count(self):return self.mfs_per_input**4
