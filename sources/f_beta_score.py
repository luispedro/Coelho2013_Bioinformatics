def f_beta_score(cmat, beta=1.):
    b2 = beta*beta
    cmat = cmat.astype(float)
    Tn,Fp,Fn,Tp = cmat.ravel()
    Fb = (1+b2)*Tp/( (1+b2)*Tp + b2*Fn + Fp)
    return Fb
