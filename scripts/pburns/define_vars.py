
def define_vars(scalar, subgrid_total_um=False, leonard=False, computeLocalDiffCoef=False):
    # Common variables:
    vars_ = {
        'm01s00i002': (r'$u$', r'm s$^{-1}$'),
        'm01s00i003': (r'$v$', r'm s$^{-1}$'),
        'm01s00i150': (r'$w$', r'm s$^{-1}$'),
        'm01s03i504': (r'$K_{\phi}$', r'kg m$^{-1}$ s$^{-1}$'),
        'm01s03i506': (r'$K_{\phi}^{\rm sf}$', r'kg m$^{-1}$ s$^{-1}$'),
        'm01s03i508': (r'$K_{\phi}^{\rm sc}$', r'kg m$^{-1}$ s$^{-1}$'),
        'm01s03i717': (r'$K_{\phi}$', r'kg m$^{-1}$ s$^{-1}$'),
        'm01s03i718': (r'$K_{\phi}^{\rm sf}$', r'kg m$^{-1}$ s$^{-1}$'),
        'm01s00i389': (r'$\rho$', r'kg m^{-3}$'),
        'm01s03i025': (r'$z_{\rm bl}$', r'm'),
        'm01s03i358': (r'$z_{\rm loc}$', r'm'),
        'm01s03i465': (r'$u_{\ast}$', r'm s$^{-1}$'),
        'm01s03i466': (r'$w_{\ast}$', r'm s$^{-1}$'),
        'm01s03i513': (r'$W_{\rm 1d}$', r''),
       }

    if scalar == 'theta':
        vars_ = {
            'm01s16i004': (r'$T$', r'K'),
            'm01s03i719': (r'$\gamma_{\phi}$', r''),
            'm01s00i004': (r'$\theta$', r'K'),
            'm01s00i010': (r'$q_v$', r'kg kg$^{-1}$'),
            'm01s00i254': (r'$q_l$', r'kg kg$^{-1}$'),
            'm01s00i012': (r'$q_f$', r'kg kg$^{-1}$'),
            'm01s03i217': (r'$\overline{w^{\prime}\,\theta^{\prime}}_0$', r'W m$^{-2}$'),
            'm01s03i467': (r'$\overline{w^{\prime}\,b}$', r''),
           }
        if subgrid_total_um == 1:
            vars_.update(
           {
            'm01s03i714': (r'$\overline{w^{\prime}\,\theta^{\prime}}_{\rm grad}$', r'W m$^{-2}$'),
            'm01s03i715': (r'$\overline{w^{\prime}\,\theta^{\prime}}_{\rm non grad}$', r'W m$^{-2}$'),
            'm01s03i716': (r'$\overline{w^{\prime}\,\theta^{\prime}}_{\rm entrain}$', r'W m$^{-2}$'),
            'm01s03i216': (r'$\overline{w^{\prime}\,\theta^{\prime}}$', r'W m$^{-2}$'),
           })
        if leonard == 1:
            vars_.update(
           {
            'm01s03i556': (r'$L_{3\theta}$', r'W m$^{-2}$'),
            'm01s03i552': (r'c_L', r''),
           })

    if scalar == 'q':
        vars_.update(
       {
        'm01s00i010': (r'$q$', r'kg kg$^{-1}$'),
        'm01s03i234': (r'$\overline{w^{\prime}\q^{\prime}}_0$', r''),
       })
        if leonard:
            vars_.update(
           {
        'm01s03i557': (r'$L_{3q}$', r'kg m$^{-2}$ s^{-1}'),
        'm01s03i552': (r'c_L', r''),
       })

    if computeLocalDiffCoef:
        vars_.update(
       {
        'm01s13i192': (r'$|S|$', r''),
        'm01s13i193': (r'$l$', r'm'),
        'm01s03i511': (r'f$_{\phi}$(Ri)', r''),
       })

    return vars_
