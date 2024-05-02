from protes import protes


def demofed():
    i_opt=np.zeros(10)
    y_opt=np.zeros(10)

    d =  5              # Dimension
    n = 11              # Mode size
    m = int(10000)       # Number of requests to the objective function
    seed = [random.randint(0, 100) for _ in range(10)]
    t=tpc()
    f1 = func_buildfed(d, n)
    f2 = func_build_alp(d, n)
    f4 = func_build_griewank(d, n)
    f5 = func_build_michalewicz(d, m)
    f8 = func_build_Rastrigin(d, m)
    f10 = func_build_Schwefel(d, n)
    functions = [f1, f2, f4, f5, f8, f10]
    y_value = ['y_values']
    t_value = ['t_values']
    for f in functions:
      for i in range(10):
          t = tpc()
          i_opt, y_optk = protes(f, d, n, m, log=True, k = 100, k_top=10, seed=seed[i])
          y_opt[i]=y_optk
          print(f'\nRESULT | y opt = {y_optk:-11.4e} | time = {tpc()-t:-10.4f}\n\n')
      y_value.append(np.min(y_opt))
      t_value.append(np.max(tpc()-t))
    return y_value, t_value

y_value, t_value = demofed()
